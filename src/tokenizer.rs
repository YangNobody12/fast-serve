//! HuggingFace Rust `tokenizers` wrapper with exact Qwen3.5 chat template & O(1) decode cache.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ChatMsg {
    pub role: String,
    pub content: String,
}

pub struct QwenTokenizer {
    inner: Tokenizer,
    eos_ids: Vec<i64>,
    single_cache: HashMap<i64, String>,
}

impl QwenTokenizer {
    pub fn load(model_dir: &Path, cfg_eos: i64) -> Result<Self> {
        let inner = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("load tokenizer.json: {e}"))?;
        let mut eos_ids = vec![cfg_eos, 248046, 248044]; // <|im_end|>, <|endoftext|>
        eos_ids.sort_unstable();
        eos_ids.dedup();
        Ok(Self {
            inner,
            eos_ids,
            single_cache: HashMap::with_capacity(4096),
        })
    }

    pub fn is_eos(&self, tid: i64) -> bool {
        self.eos_ids.contains(&tid)
    }

    /// Exact match to Qwen3.5 `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`.
    /// Returns `(full_ids, sys_len, history_len)` where:
    /// - `sys_len`: token length of `messages[0]` when `role == "system"` (reusable across conversations)
    /// - `history_len`: token length of all messages right before `<|im_start|>assistant\n<think>\n\n</think>\n\n`
    ///   (reusable across multi-turn follow-ups and regenerate!)
    pub fn format_messages(&self, messages: &[ChatMsg]) -> Result<(Vec<i64>, usize, usize)> {
        let mut full = String::new();
        let mut sys_prefix_str: Option<String> = None;

        for (i, m) in messages.iter().enumerate() {
            full.push_str("<|im_start|>");
            full.push_str(&m.role);
            full.push('\n');
            full.push_str(&m.content);
            full.push_str("<|im_end|>\n");
            if i == 0 && m.role == "system" && messages.len() >= 2 {
                sys_prefix_str = Some(full.clone());
            }
        }
        let history_str = full.clone();
        full.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n");

        let enc = self
            .inner
            .encode(full, false)
            .map_err(|e| anyhow!("encode error: {e}"))?;
        let full_ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();

        let mut sys_len = 0usize;
        if let Some(sys_s) = sys_prefix_str {
            if let Ok(senc) = self.inner.encode(sys_s, false) {
                let sids: Vec<i64> = senc.get_ids().iter().map(|&x| x as i64).collect();
                if sids.len() < full_ids.len() && full_ids[..sids.len()] == sids[..] {
                    sys_len = sids.len();
                }
            }
        }

        let mut history_len = 0usize;
        if !history_str.is_empty() {
            if let Ok(henc) = self.inner.encode(history_str, false) {
                let hids: Vec<i64> = henc.get_ids().iter().map(|&x| x as i64).collect();
                if hids.len() < full_ids.len() && full_ids[..hids.len()] == hids[..] {
                    history_len = hids.len();
                }
            }
        }

        Ok((full_ids, sys_len, history_len))
    }

    pub fn decode_single(&mut self, tid: i64) -> String {
        if let Some(s) = self.single_cache.get(&tid) {
            return s.clone();
        }
        let s = self
            .inner
            .decode(&[tid as u32], true)
            .unwrap_or_default();
        if self.single_cache.len() < 16384 {
            self.single_cache.insert(tid, s.clone());
        }
        s
    }

    pub fn decode_ids(&self, ids: &[i64]) -> String {
        let u32s: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        self.inner.decode(&u32s, true).unwrap_or_default()
    }
}

