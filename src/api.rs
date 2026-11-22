//! Axum OpenAI-compatible SSE Server + LLama-GUI static hosting (100% compatible with AIPerf & LLama-GUI).

use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;
use tower_http::cors::CorsLayer;
use tower_http::services::{ServeDir, ServeFile};
use uuid::Uuid;

use crate::batch_engine::{BatchClient, BatchSubmitReq};
use crate::engine::{RustSplitEngine, StreamToken};
use crate::tokenizer::ChatMsg;

pub enum EngineMode {
    Single(Mutex<RustSplitEngine>),
    Batched(Arc<BatchClient>),
}

pub struct AppState {
    pub mode: EngineMode,
}

#[derive(Deserialize)]
pub struct ChatRequest {
    #[serde(default = "default_model")]
    pub model: String,
    pub messages: Vec<Value>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub ignore_eos: bool,
}

fn default_model() -> String {
    "Qwen3.5-4B-split".to_string()
}

#[derive(Serialize)]
struct SseDelta<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
}

#[derive(Serialize)]
struct SseChoice<'a> {
    index: u32,
    delta: SseDelta<'a>,
    finish_reason: Option<&'static str>,
}

#[derive(Serialize)]
struct SseChunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: [SseChoice<'a>; 1],
}

#[derive(Serialize)]
struct FastStatusResp {
    engine: String,
    split_layer: usize,
    max_seq: usize,
    max_batch: usize,
    active_batch: usize,
    queued_requests: usize,
    kv_blocks_used: usize,
    kv_blocks_total: usize,
    kv_block_size: usize,
    is_loaded: bool,
    load_time_s: f64,
    prefix_cache_entries: usize,
    prefix_cache_max_slots: usize,
    prefix_cache_hits: u64,
    prefix_cache_saved_tokens: u64,
    prefix_cache_ram_mb: f64,
    mtp_drafts: usize,
    mtp_accept_rate: f64,
    mtp_avg_tokens_per_step: f64,
    vram_allocated_mib: f64,
    vram_free_mib: f64,
    vram_total_mib: f64,
}

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn parse_messages(raw: &[Value]) -> Vec<ChatMsg> {
    raw.iter()
        .map(|m| {
            let role = m.get("role").and_then(|v| v.as_str()).unwrap_or("user").to_string();
            let content = match m.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(arr)) => arr
                    .iter()
                    .filter_map(|p| {
                        if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                            p.get("text").and_then(|t| t.as_str()).map(|s| s.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => String::new(),
            };
            ChatMsg { role, content }
        })
        .collect()
}

async fn list_models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{
            "id": "Qwen3.5-4B-split",
            "object": "model",
            "created": unix_ts(),
            "owned_by": "rust-split-2part"
        }]
    }))
}

async fn fast_status(State(st): State<Arc<AppState>>) -> Json<FastStatusResp> {
    match &st.mode {
        EngineMode::Single(eng_mtx) => {
            let eng = eng_mtx.lock().unwrap();
            let (free_b, total_b) = eng.cuda.mem_info().unwrap_or((0, 0));
            let (_, _, acc_rate, avg_toks) = eng.mtp_stats();
            let (pc_entries, pc_max, pc_hits, pc_saved, pc_ram_mb) = eng.prefix_cache_stats();
            Json(FastStatusResp {
                engine: if eng.mtp_drafts() > 0 {
                    format!("Pure Rust SplitEngine + MTP (K={}) + 3-Tier Hybrid Prefix Cache", eng.mtp_drafts())
                } else {
                    "Pure Rust SplitEngine + 3-Tier Hybrid Prefix Cache (axum + Direct CUDA/cuBLAS)".to_string()
                },
                split_layer: eng.split_layer,
                max_seq: eng.max_seq,
                max_batch: 1,
                active_batch: 0,
                queued_requests: 0,
                kv_blocks_used: 0,
                kv_blocks_total: 0,
                kv_block_size: 16,
                is_loaded: true,
                load_time_s: (eng.load_time_s * 100.0).round() / 100.0,
                prefix_cache_entries: pc_entries,
                prefix_cache_max_slots: pc_max,
                prefix_cache_hits: pc_hits,
                prefix_cache_saved_tokens: pc_saved,
                prefix_cache_ram_mb: (pc_ram_mb * 10.0).round() / 10.0,
                mtp_drafts: eng.mtp_drafts(),
                mtp_accept_rate: (acc_rate * 1000.0).round() / 10.0,
                mtp_avg_tokens_per_step: (avg_toks * 100.0).round() / 100.0,
                vram_allocated_mib: (eng.vram_allocated_mib() * 10.0).round() / 10.0,
                vram_free_mib: (free_b as f64 / (1024.0 * 1024.0) * 10.0).round() / 10.0,
                vram_total_mib: (total_b as f64 / (1024.0 * 1024.0) * 10.0).round() / 10.0,
            })
        }
        EngineMode::Batched(client) => {
            let s = client.status.lock().unwrap().clone();
            Json(FastStatusResp {
                engine: format!(
                    "Pure Rust Continuous Batching (B={}) + Paged KV ({} blks x {})",
                    s.max_batch, s.kv_blocks_total, s.kv_block_size
                ),
                split_layer: s.split_layer,
                max_seq: s.max_seq,
                max_batch: s.max_batch,
                active_batch: s.active_batch,
                queued_requests: s.queued_requests,
                kv_blocks_used: s.kv_blocks_used,
                kv_blocks_total: s.kv_blocks_total,
                kv_block_size: s.kv_block_size,
                is_loaded: true,
                load_time_s: (s.load_time_s * 100.0).round() / 100.0,
                prefix_cache_entries: 0,
                prefix_cache_max_slots: 0,
                prefix_cache_hits: 0,
                prefix_cache_saved_tokens: 0,
                prefix_cache_ram_mb: 0.0,
                mtp_drafts: 0,
                mtp_accept_rate: 0.0,
                mtp_avg_tokens_per_step: 1.0,
                vram_allocated_mib: (s.vram_allocated_mib * 10.0).round() / 10.0,
                vram_free_mib: (s.vram_free_mib * 10.0).round() / 10.0,
                vram_total_mib: (s.vram_total_mib * 10.0).round() / 10.0,
            })
        }
    }
}

async fn load_fast(State(st): State<Arc<AppState>>) -> Json<FastStatusResp> {
    fast_status(State(st)).await
}

async fn chat_completions(State(st): State<Arc<AppState>>, Json(req): Json<ChatRequest>) -> Response {
    let max_new = req.max_completion_tokens.or(req.max_tokens).unwrap_or(64).clamp(1, 1024);
    let req_id = format!("chatcmpl-{}", &Uuid::new_v4().simple().to_string()[..16]);
    let created = unix_ts();
    let model_id = if req.model.is_empty() { "Qwen3.5-4B-split".to_string() } else { req.model };
    let msgs = parse_messages(&req.messages);
    let ignore_eos = req.ignore_eos;

    // ---- Phase R7: Non-Blocking Continuous Batching + Paged KV Path ----
    if let EngineMode::Batched(client) = &st.mode {
        let client = client.clone();
        let (prompt_ids, _, _) = match client.tok.format_messages(&msgs) {
            Ok(v) => v,
            Err(e) => return Json(json!({"error": e.to_string()})).into_response(),
        };
        let (tok_tx, mut tok_rx) = mpsc::unbounded_channel::<StreamToken>();
        let t_start = Instant::now();
        if client.req_tx.send(BatchSubmitReq {
            prompt_ids,
            max_new_tokens: max_new,
            ignore_eos,
            tx: tok_tx,
        }).is_err() {
            return Json(json!({"error": "Batch scheduler offline"})).into_response();
        }

        if req.stream {
            let (sse_tx, sse_rx) = mpsc::unbounded_channel::<String>();
            tokio::spawn(async move {
                let mut t_first: Option<Instant> = None;
                let mut last_prompt = 0usize;
                let mut last_comp = 0usize;
                while let Some(tok) = tok_rx.recv().await {
                    let now = Instant::now();
                    if t_first.is_none() { t_first = Some(now); }
                    last_prompt = tok.prompt_tokens;
                    last_comp = tok.completion_tokens;
                    let chunk = SseChunk {
                        id: &req_id,
                        object: "chat.completion.chunk",
                        created,
                        model: &model_id,
                        choices: [SseChoice {
                            index: 0,
                            delta: SseDelta {
                                content: if tok.text.is_empty() { None } else { Some(&tok.text) },
                            },
                            finish_reason: tok.finish_reason,
                        }],
                    };
                    if sse_tx.send(serde_json::to_string(&chunk).unwrap_or_default()).is_err() {
                        return;
                    }
                    if tok.finish_reason.is_some() { break; }
                }
                let t_end = Instant::now();
                let ttft_ms = t_first.unwrap_or(t_end).duration_since(t_start).as_secs_f64() * 1000.0;
                let dec_ms = t_end.duration_since(t_first.unwrap_or(t_end)).as_secs_f64() * 1000.0;
                let itl_ms = if last_comp > 1 { dec_ms / (last_comp - 1) as f64 } else { 0.0 };
                let tps = if itl_ms > 0.0 { 1000.0 / itl_ms } else { 0.0 };
                let usage_chunk = json!({
                    "id": req_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": last_prompt,
                        "completion_tokens": last_comp,
                        "total_tokens": last_prompt + last_comp,
                        "prompt_tokens_details": { "cached_tokens": 0 }
                    },
                    "timings": {
                        "prompt_n": last_prompt,
                        "cache_n": 0,
                        "cache_tier": "paged_kv",
                        "cache_restore_ms": 0.0,
                        "prompt_ms": (ttft_ms * 100.0).round() / 100.0,
                        "predicted_n": last_comp,
                        "predicted_ms": (dec_ms * 100.0).round() / 100.0,
                        "predicted_per_token_ms": (itl_ms * 100.0).round() / 100.0,
                        "predicted_per_second": (tps * 100.0).round() / 100.0,
                        "ttft_ms": (ttft_ms * 100.0).round() / 100.0
                    }
                });
                let _ = sse_tx.send(usage_chunk.to_string());
                let _ = sse_tx.send("[DONE]".to_string());
            });
            let stream = UnboundedReceiverStream::new(sse_rx)
                .map(|data| Ok::<Event, Infallible>(Event::default().data(data)));
            return Sse::new(stream).into_response();
        } else {
            let mut t_first: Option<Instant> = None;
            let mut text = String::new();
            let mut p_toks = 0usize;
            let mut c_toks = 0usize;
            let mut finish = "stop";
            while let Some(tok) = tok_rx.recv().await {
                if t_first.is_none() { t_first = Some(Instant::now()); }
                text.push_str(&tok.text);
                p_toks = tok.prompt_tokens;
                c_toks = tok.completion_tokens;
                if let Some(f) = tok.finish_reason {
                    finish = f;
                    break;
                }
            }
            let t_end = Instant::now();
            let ttft_ms = t_first.unwrap_or(t_end).duration_since(t_start).as_secs_f64() * 1000.0;
            let dec_ms = t_end.duration_since(t_first.unwrap_or(t_end)).as_secs_f64() * 1000.0;
            let itl_ms = if c_toks > 1 { dec_ms / (c_toks - 1) as f64 } else { 0.0 };
            return Json(json!({
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": text },
                    "finish_reason": finish
                }],
                "usage": {
                    "prompt_tokens": p_toks,
                    "completion_tokens": c_toks,
                    "total_tokens": p_toks + c_toks,
                    "prompt_tokens_details": { "cached_tokens": 0 }
                },
                "timings": {
                    "prompt_n": p_toks,
                    "cache_n": 0,
                    "cache_tier": "paged_kv",
                    "cache_restore_ms": 0.0,
                    "prompt_ms": (ttft_ms * 100.0).round() / 100.0,
                    "predicted_n": c_toks,
                    "predicted_ms": (dec_ms * 100.0).round() / 100.0,
                    "predicted_per_token_ms": (itl_ms * 100.0).round() / 100.0,
                    "predicted_per_second": if itl_ms > 0.0 { (100000.0 / itl_ms).round() / 100.0 } else { 0.0 },
                    "ttft_ms": (ttft_ms * 100.0).round() / 100.0
                }
            })).into_response();
        }
    }

    if req.stream {
        let (tx, rx) = mpsc::unbounded_channel::<String>();
        let st_clone = st.clone();

        tokio::task::spawn_blocking(move || {
            let EngineMode::Single(eng_mtx) = &st_clone.mode else { return; };
            let mut eng = eng_mtx.lock().unwrap();
            let (prompt_ids, sys_len, hist_len) = match eng.tok.format_messages(&msgs) {
                Ok(v) => v,
                Err(e) => {
                    let _ = tx.send(json!({"error": e.to_string()}).to_string());
                    return;
                }
            };

            let t_start = Instant::now();
            let mut t_first: Option<Instant> = None;
            let mut last_prompt = 0usize;
            let mut last_cached = 0usize;
            let mut last_tier: &'static str = "miss";
            let mut last_restore_ms: f64 = 0.0;
            let mut last_comp = 0usize;

            let res = eng.generate_stream(&prompt_ids, sys_len, hist_len, max_new, ignore_eos, |tok| {
                let now = Instant::now();
                if t_first.is_none() {
                    t_first = Some(now);
                }
                last_prompt = tok.prompt_tokens;
                last_cached = tok.cached_tokens;
                last_tier = tok.cache_tier;
                last_restore_ms = tok.cache_restore_ms;
                last_comp = tok.completion_tokens;

                let chunk = SseChunk {
                    id: &req_id,
                    object: "chat.completion.chunk",
                    created,
                    model: &model_id,
                    choices: [SseChoice {
                        index: 0,
                        delta: SseDelta {
                            content: if tok.text.is_empty() { None } else { Some(&tok.text) },
                        },
                        finish_reason: tok.finish_reason,
                    }],
                };
                let serialized = serde_json::to_string(&chunk).unwrap_or_default();
                tx.send(serialized).is_ok()
            });

            if let Err(e) = res {
                let _ = tx.send(json!({"error": e.to_string()}).to_string());
                return;
            }

            let t_end = Instant::now();
            let ttft_ms = t_first.unwrap_or(t_end).duration_since(t_start).as_secs_f64() * 1000.0;
            let dec_ms = t_end.duration_since(t_first.unwrap_or(t_end)).as_secs_f64() * 1000.0;
            let itl_ms = if last_comp > 1 { dec_ms / (last_comp - 1) as f64 } else { 0.0 };
            let tps = if itl_ms > 0.0 { 1000.0 / itl_ms } else { 0.0 };

            let usage_chunk = json!({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [],
                "usage": {
                    "prompt_tokens": last_prompt,
                    "completion_tokens": last_comp,
                    "total_tokens": last_prompt + last_comp,
                    "prompt_tokens_details": { "cached_tokens": last_cached }
                },
                "timings": {
                    "prompt_n": last_prompt,
                    "cache_n": last_cached,
                    "cache_tier": last_tier,
                    "cache_restore_ms": (last_restore_ms * 100.0).round() / 100.0,
                    "prompt_ms": (ttft_ms * 100.0).round() / 100.0,
                    "predicted_n": last_comp,
                    "predicted_ms": (dec_ms * 100.0).round() / 100.0,
                    "predicted_per_token_ms": (itl_ms * 100.0).round() / 100.0,
                    "predicted_per_second": (tps * 100.0).round() / 100.0,
                    "ttft_ms": (ttft_ms * 100.0).round() / 100.0
                }
            });
            let _ = tx.send(usage_chunk.to_string());
            let _ = tx.send("[DONE]".to_string());
        });

        let stream = UnboundedReceiverStream::new(rx)
            .map(|data| Ok::<Event, Infallible>(Event::default().data(data)));
        Sse::new(stream).into_response()
    } else {
        let st_clone = st.clone();
        let out = tokio::task::spawn_blocking(move || {
            let EngineMode::Single(eng_mtx) = &st_clone.mode else { unreachable!() };
            let mut eng = eng_mtx.lock().unwrap();
            let (prompt_ids, sys_len, hist_len) = eng.tok.format_messages(&msgs).unwrap();
            let t_start = Instant::now();
            let mut t_first: Option<Instant> = None;
            let mut text = String::new();
            let mut p_toks = 0;
            let mut c_toks = 0;
            let mut cached = 0;
            let mut tier: &'static str = "miss";
            let mut restore_ms: f64 = 0.0;
            let mut finish = "stop";
            let _ = eng.generate_stream(&prompt_ids, sys_len, hist_len, max_new, ignore_eos, |tok| {
                if t_first.is_none() {
                    t_first = Some(Instant::now());
                }
                text.push_str(&tok.text);
                p_toks = tok.prompt_tokens;
                c_toks = tok.completion_tokens;
                cached = tok.cached_tokens;
                tier = tok.cache_tier;
                restore_ms = tok.cache_restore_ms;
                if let Some(f) = tok.finish_reason {
                    finish = f;
                }
                true
            });
            let t_end = Instant::now();
            let ttft_ms = t_first.unwrap_or(t_end).duration_since(t_start).as_secs_f64() * 1000.0;
            let dec_ms = t_end.duration_since(t_first.unwrap_or(t_end)).as_secs_f64() * 1000.0;
            let itl_ms = if c_toks > 1 { dec_ms / (c_toks - 1) as f64 } else { 0.0 };
            json!({
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": text },
                    "finish_reason": finish
                }],
                "usage": {
                    "prompt_tokens": p_toks,
                    "completion_tokens": c_toks,
                    "total_tokens": p_toks + c_toks,
                    "prompt_tokens_details": { "cached_tokens": cached }
                },
                "timings": {
                    "prompt_n": p_toks,
                    "cache_n": cached,
                    "cache_tier": tier,
                    "cache_restore_ms": (restore_ms * 100.0).round() / 100.0,
                    "prompt_ms": (ttft_ms * 100.0).round() / 100.0,
                    "predicted_n": c_toks,
                    "predicted_ms": (dec_ms * 100.0).round() / 100.0,
                    "predicted_per_token_ms": (itl_ms * 100.0).round() / 100.0,
                    "predicted_per_second": if itl_ms > 0.0 { (100000.0 / itl_ms).round() / 100.0 } else { 0.0 },
                    "ttft_ms": (ttft_ms * 100.0).round() / 100.0
                }
            })
        })
        .await
        .unwrap();
        Json(out).into_response()
    }
}

pub fn build_router(state: Arc<AppState>, static_dir: PathBuf) -> Router {
    let chat_html = static_dir.join("chat.html");
    let vis_html = static_dir.join("index.html");
    Router::new()
        .route_service("/", ServeFile::new(chat_html.clone()))
        .route_service("/chat", ServeFile::new(chat_html))
        .route_service("/3d", ServeFile::new(vis_html))
        .nest_service("/static", ServeDir::new(static_dir))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/api/fast_status", get(fast_status))
        .route("/api/load_fast", post(load_fast))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

