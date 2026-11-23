mod api;
mod batch_engine;
mod config;
mod cuda_ffi;
mod engine;
mod kv_pool;
mod loader;
mod tokenizer;

use anyhow::Result;
use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::batch_engine::{BatchClient, BatchGpuEngine};
use crate::engine::{PrefillInt8Mode, RustSplitEngine};
use crate::tokenizer::ChatMsg;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();
    let mut model_dir = PathBuf::from(r"D:\serving\Qwen3.5-4B");
    let mut static_dir = PathBuf::from(r"D:\serving\static");
    let mut port: u16 = 8000;
    let mut split_layer: usize = 16;
    let mut max_seq: usize = 2048;
    let mut max_batch: usize = 1;
    let mut mtp_drafts: usize = 3;
    let mut prefix_cache_slots: usize = 8;
    let mut force_max_seq = false;
    let mut skinny_m4_wide = true;
    let mut prefill_int8 = PrefillInt8Mode::Smooth;
    let mut test_only = false;
    let mut profile_nsys = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-dir" => { i += 1; model_dir = PathBuf::from(&args[i]); }
            "--static-dir" => { i += 1; static_dir = PathBuf::from(&args[i]); }
            "--port" => { i += 1; port = args[i].parse()?; }
            "--split-layer" => { i += 1; split_layer = args[i].parse()?; }
            "--max-seq" => { i += 1; max_seq = args[i].parse()?; }
            "--max-batch" => { i += 1; max_batch = args[i].parse()?; }
            "--mtp-drafts" => { i += 1; mtp_drafts = args[i].parse()?; }
            "--prefix-cache-slots" => { i += 1; prefix_cache_slots = args[i].parse()?; }
            "--force-max-seq" => { force_max_seq = true; }
            "--no-skinny-m4-wide" => { skinny_m4_wide = false; }
            "--prefill-int8" => {
                i += 1;
                prefill_int8 = match args[i].as_str() {
                    "off" | "bf16" | "0" => PrefillInt8Mode::Off,
                    "smooth" | "1" => PrefillInt8Mode::Smooth,
                    "ozaki" | "2" => PrefillInt8Mode::Ozaki,
                    other => anyhow::bail!("Unknown --prefill-int8 mode: {other} (use off|smooth|ozaki)"),
                };
            }
            "--test-only" => { test_only = true; }
            "--profile-nsys" => { profile_nsys = true; }
            _ => {}
        }
        i += 1;
    }

    // ---- Phase R7: Continuous Batching + Paged KV Block Manager Mode (--max-batch 2..4) ----
    if max_batch > 1 {
        if profile_nsys {
            use std::thread::sleep;
            use std::time::Duration;
            println!("[BatchEngine] Profiling Continuous Batching (B={max_batch}, max_seq={max_seq}, prefill_int8={prefill_int8:?})...");
            let mut beng = BatchGpuEngine::load(&model_dir, split_layer, max_seq, max_batch, force_max_seq, prefill_int8)?;
            let p_base = "Explain GPU inference serving architecture, continuous batching, paged KV cache, and Tensor Core GEMMs. ";
            let (mut ids128, _, _) = beng.tok.format_messages(&[ChatMsg { role: "user".into(), content: p_base.repeat(6) }])?;
            ids128.truncate(128);
            // Warmup
            let _ = beng.profile_batch_decode_64(&ids128)?;
            sleep(Duration::from_millis(60));
            beng.cuda.profiler_start()?;
            sleep(Duration::from_millis(40));
            let dec_ms = beng.profile_batch_decode_64(&ids128)?;
            sleep(Duration::from_millis(40));
            beng.cuda.profiler_stop()?;
            let step_ms = dec_ms / 64.0;
            println!(
                "[BatchProfile] B={} DECODE_64: {:.2} ms | {:.2} ms/step ({:.1} tok/s/user · {:.1} tok/s TOTAL!)",
                beng.max_batch,
                dec_ms,
                step_ms,
                1000.0 / step_ms,
                (beng.max_batch as f64) * 1000.0 / step_ms,
            );
            return Ok(());
        }

        println!("[BatchEngine] Starting Continuous Batching + Paged KV Server (max_batch={max_batch}, max_seq={max_seq}, prefill_int8={prefill_int8:?})...");
        let client = BatchClient::start(&model_dir, split_layer, max_seq, max_batch, force_max_seq, prefill_int8)?;
        {
            let s = client.status.lock().unwrap();
            println!(
                "[BatchEngine] Loaded in {:.2}s | MaxBatch={} | Ctx={} | PagedKV={} blks ({} pooled tok) | VRAM alloc={:.1} MiB | free={:.1} MiB / {:.1} MiB",
                s.load_time_s, s.max_batch, s.max_seq, s.kv_blocks_total, s.kv_blocks_total * s.kv_block_size,
                s.vram_allocated_mib, s.vram_free_mib, s.vram_total_mib
            );
        }
        let state = Arc::new(api::AppState {
            mode: api::EngineMode::Batched(client),
        });
        let app = api::build_router(state, static_dir);
        let addr = format!("127.0.0.1:{port}");
        use axum::serve::ListenerExt;
        let listener = tokio::net::TcpListener::bind(&addr)
            .await?
            .tap_io(|tcp_stream| {
                let _ = tcp_stream.set_nodelay(true);
            });
        println!("[BatchEngine] Listening on http://{addr} (Continuous Batching B={max_batch} · Paged KV · TCP_NODELAY)");
        axum::serve(listener, app).await?;
        return Ok(());
    }

    println!("[RustSplitServer] Loading Qwen3.5-4B (split_layer={split_layer}, max_seq={max_seq}, mtp_drafts={mtp_drafts}, prefix_cache_slots={prefix_cache_slots}, skinny_m4_wide={skinny_m4_wide}, prefill_int8={prefill_int8:?})...");
    let mut eng = RustSplitEngine::load(&model_dir, split_layer, max_seq, mtp_drafts, prefix_cache_slots, force_max_seq, skinny_m4_wide, prefill_int8)?;
    let (free_b, total_b) = eng.cuda.mem_info()?;
    println!(
        "[RustSplitServer] Loaded in {:.2}s | Ctx={} | MTP K={} | PrefixSlots={} | VRAM alloc={:.1} MiB | free={:.1} MiB / {:.1} MiB",
        eng.load_time_s,
        eng.max_seq,
        eng.mtp_drafts(),
        prefix_cache_slots,
        eng.vram_allocated_mib(),
        free_b as f64 / 1048576.0,
        total_b as f64 / 1048576.0,
    );

    if profile_nsys {
        use std::thread::sleep;
        use std::time::Duration;

        // Build realistic 128-tok and 512-tok prompts so MTP speculative acceptance matches production!
        let p_base = "Explain GPU inference serving architecture, pipeline parallelism, Tensor Core GEMM utilization, memory bandwidth bottlenecks, and multi-token speculative decoding in detail. ";
        let (mut ids128, _, _) = eng.tok.format_messages(&[ChatMsg { role: "user".into(), content: p_base.repeat(6) }])?;
        ids128.truncate(128);
        let (mut ids512, _, _) = eng.tok.format_messages(&[ChatMsg { role: "user".into(), content: p_base.repeat(24) }])?;
        ids512.truncate(512);

        // Warmup before capture
        let _ = eng.prefill_fused(&ids128, 0, 0, false)?;
        let _ = eng.prefill_fused(&ids512, 0, 0, false)?;
        eng.generate_stream(&ids128, 0, 0, 8, true, |_| true)?;
        eng.cuda.sync()?;
        sleep(Duration::from_millis(60));

        // Single continuous nsys capture window with 3 clean bursts separated by 70ms idle gaps:
        // Burst 1: PREFILL_512 -> Burst 2: PREFILL_128 -> Burst 3: DECODE_64 (MTP K=3 or M=1)
        eng.cuda.profiler_start()?;
        sleep(Duration::from_millis(40));

        let t0 = Instant::now();
        let _ = eng.prefill_fused(&ids512, 0, 0, false)?;
        eng.cuda.sync()?;
        let p512_ms = t0.elapsed().as_secs_f64() * 1000.0;
        sleep(Duration::from_millis(70));

        let t0 = Instant::now();
        let tok128 = eng.prefill_fused(&ids128, 0, 0, false)?.first_tok;
        eng.cuda.sync()?;
        let p128_ms = t0.elapsed().as_secs_f64() * 1000.0;
        sleep(Duration::from_millis(70));

        let (dec_total, avg_dec, extra) = if eng.mtp_drafts() > 0 {
            let (cycles, ms) = eng.decode_64_mtp_profiled(&ids128, tok128)?;
            let (acc, drf, rate, avg_t) = eng.mtp_stats();
            (ms, ms / 64.0, format!(" | MTP K={} cycles={cycles} accept={acc}/{drf} ({:.1}%) avg={avg_t:.2} tok/step", eng.mtp_drafts(), rate * 100.0))
        } else {
            let t_dec = Instant::now();
            let mut step_ms = Vec::with_capacity(64);
            eng.decode_64_profiled(tok128, 128, &mut step_ms)?;
            let ms = t_dec.elapsed().as_secs_f64() * 1000.0;
            let avg = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
            (ms, avg, String::new())
        };
        sleep(Duration::from_millis(40));
        eng.cuda.profiler_stop()?;

        println!("[RustProfile] PREFILL_128: {p128_ms:.2} ms");
        println!("[RustProfile] PREFILL_512: {p512_ms:.2} ms");
        println!("[RustProfile] DECODE_64  : {dec_total:.2} ms | avg={avg_dec:.2} ms/tok ({:.1} tok/s){extra}", 1000.0 / avg_dec);
        return Ok(());
    }

    if test_only {
        // Phase R1-R4 bit-exactness & speed verification!
        let msgs = vec![ChatMsg {
            role: "user".into(),
            content: "อธิบายการแยกโมเดล AI เป็น 2 ส่วนแบบสั้นๆ".into(),
        }];
        let (pids, slen, hlen) = eng.tok.format_messages(&msgs)?;
        let mut out_ids = Vec::new();
        eng.generate_stream(&pids, slen, hlen, 10, true, |t| {
            out_ids.push(t.token_id);
            true
        })?;
        println!("[Test] THAI OUT (10 tok): {}", eng.tok.decode_ids(&out_ids));

        // 64-token speed test
        let bench_msgs = vec![ChatMsg {
            role: "user".into(),
            content: "Explain pipeline parallelism and GPU inference optimization. ".repeat(6),
        }];
        let (bpids, bslen, bhlen) = eng.tok.format_messages(&bench_msgs)?;
        for run in 1..=4 {
            let t0 = Instant::now();
            let mut times = Vec::with_capacity(64);
            let mut c_toks = 0usize;
            let mut c_tier = "miss";
            let mut c_ms = 0.0f64;
            eng.generate_stream(&bpids, bslen, bhlen, 64, true, |t| {
                times.push(Instant::now());
                c_toks = t.cached_tokens;
                c_tier = t.cache_tier;
                c_ms = t.cache_restore_ms;
                true
            })?;
            let ttft = times[0].duration_since(t0).as_secs_f64() * 1000.0;
            let itl = times.last().unwrap().duration_since(times[0]).as_secs_f64() * 1000.0
                / (times.len() - 1) as f64;
            println!(
                "[Test] Run {run} ({} in, cached={c_toks} [{c_tier} {c_ms:.2}ms] -> 64 out): TTFT={ttft:.2} ms | ITL={itl:.2} ms/tok ({:.1} tok/s)",
                bpids.len(),
                1000.0 / itl
            );
        }
        let (acc, drf, rate, avg_t) = eng.mtp_stats();
        if drf > 0 {
            println!(
                "[Test] MTP Stats (K={}): accepted={acc}/{drf} ({:.1}%) | avg={avg_t:.2} tok/verify-step",
                eng.mtp_drafts(),
                rate * 100.0
            );
        }
        return Ok(());
    }

    let state = Arc::new(api::AppState {
        mode: api::EngineMode::Single(Mutex::new(eng)),
    });
    let app = api::build_router(state, static_dir);
    let addr = format!("127.0.0.1:{port}");
    use axum::serve::ListenerExt;
    let listener = tokio::net::TcpListener::bind(&addr)
        .await?
        .tap_io(|tcp_stream| {
            let _ = tcp_stream.set_nodelay(true);
        });
    println!("[RustSplitServer] Listening on http://{addr} (TCP_NODELAY · LLama-GUI + OpenAI /v1/chat/completions)");
    axum::serve(listener, app).await?;
    Ok(())
}

