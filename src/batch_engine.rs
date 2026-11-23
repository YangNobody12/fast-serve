//! Phase R7: Iteration-Level Continuous Batching Engine + Paged KV Block Manager (B = 1..4).
//! Shares ONE 8.0 GB weight read across up to 4 concurrent users (~240-260 tok/s total on RTX 3080)!

use anyhow::Result;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::path::Path;
use std::sync::{mpsc as std_mpsc, Arc, Mutex};
use std::thread;
use std::time::Instant;
use tokio::sync::mpsc as tokio_mpsc;

use crate::config::TextConfig;
use crate::cuda_ffi::{CUdeviceptr, CUfunction, CUgraphExec, CudaRuntime};
use crate::engine::{
    build_rope_tables_bf16, launch_embed_prefill_bf12, launch_lm_head_bf12_argmax,
    launch_skinny_bf12, run_deltanet_prefill_wy, run_prefill_fused_tiled_mlp_i8,
    run_prefill_gemm_i8, Bf12DecodeFuncs, Int8PrefillFuncs, PrefillInt8Mode, PrefillWorkspace,
    StreamToken, WyKernelFuncs, PF_HEADER_ROWS, PF_POOL_BYTES, PTX_BYTES, W_UNPACK_BYTES,
};
use crate::kv_pool::{DeltaNetSlotPool, KvBlockPool, KV_BLOCK_BYTES, KV_BLOCK_SIZE, MAX_BLOCKS_PER_SEQ};
use crate::loader::{load_split_weights, Bf12Ptr, LoadedSplitWeights, PackedLayerPtrs};
use crate::tokenizer::QwenTokenizer;

struct BatchKernelFuncs {
    rmsnorm_2d: CUfunction,
    add_rmsnorm_2d: CUfunction,
    silu_mul_packed_2d: CUfunction,
    residual_add: CUfunction,
    copy_bf16: CUfunction,
    conv1d_prefill: CUfunction,
    deltanet_prefill: CUfunction,
    deltanet_prefill_norm_gate: CUfunction,
    argmax_stage2: CUfunction,
    argmax_batch_stage2: CUfunction,
    gqa_verify_m_reduce: CUfunction,
    // Phase R7: Batched & Paged KV kernels
    conv1d_batch_decode: CUfunction,
    deltanet_batch_decode: CUfunction,
    qk_rope_paged_prefill: CUfunction,
    prefill_gqa_paged: CUfunction,
    qk_rope_paged_batch_decode: CUfunction,
    paged_gqa_flash_splitk: CUfunction,
    // Phase R8: Chunked WY-Form DeltaNet Prefill
    wy: WyKernelFuncs,
    // Phase R9: 2D Parallel Conv1D Prefill
    conv1d_prefill_2d: CUfunction,
    conv1d_update_state: CUfunction,
    // Phase R10 + R11 + R12: SmoothQuant PTX-IMMA + Register-Resident 32x16 Paged FA2
    r10: Int8PrefillFuncs,
    prefill_gqa_paged_wmma_32x16: CUfunction,
    // Phase R14: Lossless BF12 Embed + Skinny Decode + Batched Fused lm_head_argmax (B=1..4)
    bf12: Bf12DecodeFuncs,
}

impl BatchKernelFuncs {
    fn load(cuda: &CudaRuntime) -> Result<Self> {
        Ok(Self {
            rmsnorm_2d: cuda.get_func("qwen35_rmsnorm_2d_bf16")?,
            add_rmsnorm_2d: cuda.get_func("qwen35_add_rmsnorm_2d_bf16")?,
            silu_mul_packed_2d: cuda.get_func("silu_mul_packed_2d_bf16")?,
            residual_add: cuda.get_func("residual_add_bf16")?,
            copy_bf16: cuda.get_func("copy_bf16")?,
            conv1d_prefill: cuda.get_func("causal_conv1d_prefill_k4_bf16")?,
            deltanet_prefill: cuda.get_func("gated_deltanet_prefill_bf16")?,
            deltanet_prefill_norm_gate: cuda.get_func("deltanet_prefill_norm_gate_bf16")?,
            argmax_stage2: cuda.get_func("argmax_stage2_bf16")?,
            argmax_batch_stage2: cuda.get_func("argmax_batch_stage2_bf16")?,
            gqa_verify_m_reduce: cuda.get_func("gqa_verify_m_flash_reduce_bf16")?,
            conv1d_batch_decode: cuda.get_func("causal_conv1d_batch_decode_bf16")?,
            deltanet_batch_decode: cuda.get_func("gated_deltanet_batch_decode_bf16")?,
            qk_rope_paged_prefill: cuda.get_func("qk_rope_paged_kv_prefill_bf16")?,
            prefill_gqa_paged: cuda.get_func("prefill_gqa_paged_causal_gated_bf16")?,
            qk_rope_paged_batch_decode: cuda.get_func("qk_rope_paged_kv_batch_decode_bf16")?,
            paged_gqa_flash_splitk: cuda.get_func("paged_gqa_decode_flash_splitk_bf16")?,
            wy: WyKernelFuncs::load(cuda)?,
            conv1d_prefill_2d: cuda.get_func("causal_conv1d_prefill_2d_bf16")?,
            conv1d_update_state: cuda.get_func("causal_conv1d_update_state_bf16")?,
            r10: Int8PrefillFuncs::load(cuda)?,
            prefill_gqa_paged_wmma_32x16: cuda.get_func("prefill_gqa_paged_wmma_32x16_causal_gated_bf16")?,
            bf12: Bf12DecodeFuncs::load(cuda)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct BatchStatusSnapshot {
    pub split_layer: usize,
    pub max_seq: usize,
    pub max_batch: usize,
    pub active_batch: usize,
    pub queued_requests: usize,
    pub kv_blocks_used: usize,
    pub kv_blocks_total: usize,
    pub kv_block_size: usize,
    pub load_time_s: f64,
    pub vram_allocated_mib: f64,
    pub vram_free_mib: f64,
    pub vram_total_mib: f64,
}

pub struct BatchSubmitReq {
    pub prompt_ids: Vec<i64>,
    pub max_new_tokens: usize,
    pub ignore_eos: bool,
    pub tx: tokio_mpsc::UnboundedSender<StreamToken>,
}

struct ActiveSeq {
    slot_id: usize,
    blocks: Vec<i32>,
    prompt_len: usize,
    pos: usize,          // next token position to decode (starts at prompt_len)
    cur_tok: i64,        // last generated token (input to next decode step)
    emitted: usize,      // tokens emitted so far (starts at 1 after prefill)
    max_new_tokens: usize,
    ignore_eos: bool,
    tx: tokio_mpsc::UnboundedSender<StreamToken>,
}

pub struct BatchGpuEngine {
    pub cuda: CudaRuntime,
    funcs: BatchKernelFuncs,
    pub cfg: TextConfig,
    pub tok: QwenTokenizer,
    pub split_layer: usize,
    pub max_seq: usize,
    pub max_batch: usize,
    pub load_time_s: f64,

    weights: LoadedSplitWeights,
    kv_pool: KvBlockPool,
    dn_pool: DeltaNetSlotPool,

    // Precomputed linear_idx (0..23) and fa_idx (0..7) per layer in p1 & p2
    p1_sub_idx: Vec<usize>,
    p2_sub_idx: Vec<usize>,

    // RoPE tables
    _rope_arena: CUdeviceptr,
    cos_table: CUdeviceptr,
    sin_table: CUdeviceptr,

    // Batched Decode Scratch Arena (B_max <= 4)
    _scratch_arena: CUdeviceptr,
    scratch_bytes: usize,
    d_tok_ids: CUdeviceptr,      // [4] i64
    d_out_toks: CUdeviceptr,     // [4] i64
    d_positions: CUdeviceptr,    // [4] i32
    d_slot_ids: CUdeviceptr,     // [4] i32
    d_block_tables: CUdeviceptr, // [4, MAX_BLOCKS_PER_SEQ] i32
    d_hidden: CUdeviceptr,       // [4, 2560] bf16
    d_bridge: CUdeviceptr,       // [4, 2560] bf16
    d_normed: CUdeviceptr,       // [4, 2560] bf16
    d_delta: CUdeviceptr,        // [4, 2560] bf16
    d_dn_in: CUdeviceptr,        // [4, 12352] bf16
    d_qkv_conv: CUdeviceptr,     // [4, 8192] bf16
    d_core_out: CUdeviceptr,     // [4, 4096] bf16
    d_fa_qkv: CUdeviceptr,       // [4, 10240] bf16
    d_q_rot: CUdeviceptr,        // [4, 16, 256] bf16
    d_flash_out: CUdeviceptr,    // [4, 16, 4, 256] f32
    d_flash_ml: CUdeviceptr,     // [4, 16, 4, 2] f32
    d_mlp_gu: CUdeviceptr,       // [4, 18432] bf16
    d_mlp_act: CUdeviceptr,      // [4, 9216] bf16
    d_arg_vals: CUdeviceptr,     // [4, 1024] f32 (Phase R14: d_logits freed!)
    d_arg_idxs: CUdeviceptr,     // [4, 1024] i32
    h_out_toks: *mut i64,        // [4] pinned host
    // Phase R9: Single contiguous pinned host meta buffer (eliminates 7x pageable WDDM HtoD stalls!)
    h_meta_pinned: *mut u8,
    meta_bytes: usize,
    meta_off_pos: usize,
    meta_off_slt: usize,
    meta_off_bt: usize,

    // Pre-allocated 1024-token Super-Chunk Prefill Workspace + shared 90.0 MiB BF12 unpack buffer
    pf_arena: CUdeviceptr,
    pf_bytes: usize,
    pf_ws: PrefillWorkspace,
    w_unpack_buf: CUdeviceptr,
    d_pf_bt: CUdeviceptr,        // [MAX_BLOCKS_PER_SEQ] i32

    // Phase R7.1: Pre-captured Batched Decode CUDA Graphs for B = 1..=max_batch (1 launch/step!)
    decode_graphs: Vec<CUgraphExec>,
    // Phase R10: Online SmoothQuant / Ozaki INT8 Tensor Core Prefill mode
    prefill_int8: PrefillInt8Mode,
}

unsafe impl Send for BatchGpuEngine {}

impl BatchGpuEngine {
    pub fn load(
        model_dir: &Path,
        split_layer: usize,
        max_seq: usize,
        max_batch: usize,
        force_max_seq: bool,
        prefill_int8: PrefillInt8Mode,
    ) -> Result<Self> {
        let t0 = Instant::now();
        let max_batch = max_batch.clamp(1, 4);
        let max_seq = max_seq.clamp(256, MAX_BLOCKS_PER_SEQ * KV_BLOCK_SIZE);
        let cfg = TextConfig::load(model_dir)?;
        let tok = QwenTokenizer::load(model_dir, cfg.eos_token_id)?;
        let cuda = CudaRuntime::new(PTX_BYTES)?;
        let funcs = BatchKernelFuncs::load(&cuda)?;

        // Load pure BF16 core weights (L0..31, no MTP weights in Continuous Batching mode -> saves 230 MiB VRAM!)
        let weights = load_split_weights(&cuda, model_dir, &cfg, split_layer, false)?;

        // Compute linear/full-attn sub-indices across Part 1 & Part 2
        let mut lin_c = 0usize;
        let mut fa_c = 0usize;
        let mut p1_sub_idx = Vec::with_capacity(weights.p1_layers.len());
        for l in &weights.p1_layers {
            if l.is_linear { p1_sub_idx.push(lin_c); lin_c += 1; } else { p1_sub_idx.push(fa_c); fa_c += 1; }
        }
        let mut p2_sub_idx = Vec::with_capacity(weights.p2_layers.len());
        for l in &weights.p2_layers {
            if l.is_linear { p2_sub_idx.push(lin_c); lin_c += 1; } else { p2_sub_idx.push(fa_c); fa_c += 1; }
        }

        // Auto WDDM VRAM Headroom Guard for Paged KV Pool!
        let (free_b, _) = cuda.mem_info()?;
        let free_mib = free_b as f64 / 1048576.0;
        let dn_mib = (max_batch as f64) * 49.5;
        let fixed_mib = dn_mib + 56.0 + 90.0 + 12.0 + 1.0; // DeltaNet + pf_arena (55.6) + w_unpack (90) + scratch + RoPE
        let wddm_guard_mib = 380.0;
        // Desired pooled blocks: enough for 4096 tokens (256 blocks = 128 MiB) or capped to stay WDDM-safe!
        let desired_blocks = (max_seq.max(2048) * 2 / KV_BLOCK_SIZE).clamp(128, 256);
        let num_blocks = if force_max_seq {
            desired_blocks
        } else {
            let avail_kv_mib = (free_mib - fixed_mib - wddm_guard_mib).max(64.0);
            let max_safe_blocks = (avail_kv_mib / (KV_BLOCK_BYTES as f64 / 1048576.0)) as usize;
            let chosen = desired_blocks.min(max_safe_blocks).max(96);
            if chosen < desired_blocks {
                println!(
                    "[BatchEngine] WDDM VRAM Guard: free={:.1} MiB after weights -> sized Paged KV Pool to {} blocks ({} pooled tokens, {:.1} MiB)",
                    free_mib, chosen, chosen * KV_BLOCK_SIZE, (chosen * KV_BLOCK_BYTES) as f64 / 1048576.0
                );
            }
            chosen
        };
        let eff_max_seq = max_seq.min(num_blocks * KV_BLOCK_SIZE);

        let dn_pool = DeltaNetSlotPool::alloc(&cuda, max_batch)?;
        let kv_pool = KvBlockPool::alloc(&cuda, num_blocks)?;

        // Upload RoPE tables [eff_max_seq, 64] bf16
        let (cos_h, sin_h) = build_rope_tables_bf16(&cfg, eff_max_seq);
        let tbl_bytes = cos_h.len() * 2;
        let tbl_aligned = (tbl_bytes + 255) & !255;
        let rope_arena = cuda.alloc(tbl_aligned * 2)?;
        let cos_table = rope_arena;
        let sin_table = rope_arena + tbl_aligned as u64;
        cuda.htod_sync(cos_table, cos_h.as_ptr() as *const u8, tbl_bytes)?;
        cuda.htod_sync(sin_table, sin_h.as_ptr() as *const u8, tbl_bytes)?;

        // Batched Decode Scratch Arena (B_max = 4)
        let align = |b: usize| (b + 255) & !255;
        let mut off = 0usize;
        let mut next = |b: usize| {
            let cur = off;
            off += align(b);
            cur as u64
        };
        // Pack [tok_ids | positions | slot_ids | block_tables] contiguously at offset 0 for 1 pinned HtoD!
        let o_tok = next(4 * 8);
        let o_pos = next(4 * 4);
        let o_slt = next(4 * 4);
        let o_bt  = next(4 * MAX_BLOCKS_PER_SEQ * 4);
        let o_out = next(4 * 8);
        let meta_bytes = o_out as usize;
        let o_hid = next(4 * 2560 * 2);
        let o_br  = next(4 * 2560 * 2);
        let o_nrm = next(4 * 2560 * 2);
        let o_del = next(4 * 2560 * 2);
        let o_dn  = next(4 * 12352 * 2);
        let o_cv  = next(4 * 8192 * 2);
        let o_co  = next(4 * 4096 * 2);
        let o_fa  = next(4 * 10240 * 2);
        let o_qr  = next(4 * 16 * 256 * 2);
        let o_fo  = next(4 * 16 * 4 * 256 * 4);
        let o_fm  = next(4 * 16 * 4 * 2 * 4);
        let o_gu  = next(4 * 18432 * 2);
        let o_ac  = next(4 * 9216 * 2);
        // Phase R14: d_logits freed (-1.9 MiB) — Batch Decode uses lm_head_bf12_argmax_m{1..4}!
        let o_av  = next(4 * 1024 * 4);
        let o_ai  = next(4 * 1024 * 4);
        let scratch_arena = cuda.alloc(off)?;
        cuda.memset_zero_async(scratch_arena, off)?;
        let h_out_toks = cuda.alloc_pinned(4 * 8)? as *mut i64;
        let h_meta_pinned = cuda.alloc_pinned(meta_bytes)?;
        unsafe { std::ptr::write_bytes(h_meta_pinned, 0, meta_bytes); }

        // Pre-allocate 1024-token Super-Chunk Prefill Workspace + shared 90.0 MiB BF12 unpack buffer
        let (pf_arena, pf_ws_bytes, pf_ws) = PrefillWorkspace::alloc(&cuda)?;
        let w_unpack_buf = cuda.alloc(W_UNPACK_BYTES)?;
        let bt_bytes = align(MAX_BLOCKS_PER_SEQ * 4);
        let d_pf_bt = cuda.alloc(bt_bytes)?;
        let poff = pf_ws_bytes + W_UNPACK_BYTES + bt_bytes;
        cuda.sync()?;

        let mut eng = Self {
            cuda,
            funcs,
            cfg,
            tok,
            split_layer,
            max_seq: eff_max_seq,
            max_batch,
            load_time_s: t0.elapsed().as_secs_f64(),
            weights,
            kv_pool,
            dn_pool,
            p1_sub_idx,
            p2_sub_idx,
            _rope_arena: rope_arena,
            cos_table,
            sin_table,
            _scratch_arena: scratch_arena,
            scratch_bytes: off,
            d_tok_ids: scratch_arena + o_tok,
            d_out_toks: scratch_arena + o_out,
            d_positions: scratch_arena + o_pos,
            d_slot_ids: scratch_arena + o_slt,
            d_block_tables: scratch_arena + o_bt,
            d_hidden: scratch_arena + o_hid,
            d_bridge: scratch_arena + o_br,
            d_normed: scratch_arena + o_nrm,
            d_delta: scratch_arena + o_del,
            d_dn_in: scratch_arena + o_dn,
            d_qkv_conv: scratch_arena + o_cv,
            d_core_out: scratch_arena + o_co,
            d_fa_qkv: scratch_arena + o_fa,
            d_q_rot: scratch_arena + o_qr,
            d_flash_out: scratch_arena + o_fo,
            d_flash_ml: scratch_arena + o_fm,
            d_mlp_gu: scratch_arena + o_gu,
            d_mlp_act: scratch_arena + o_ac,
            d_arg_vals: scratch_arena + o_av,
            d_arg_idxs: scratch_arena + o_ai,
            h_out_toks,
            h_meta_pinned,
            meta_bytes,
            meta_off_pos: o_pos as usize,
            meta_off_slt: o_slt as usize,
            meta_off_bt: o_bt as usize,
            pf_arena,
            pf_bytes: poff,
            pf_ws,
            w_unpack_buf,
            d_pf_bt,
            decode_graphs: Vec::with_capacity(max_batch),
            prefill_int8,
        };
        eng.capture_batch_decode_graphs()?;
        eng.load_time_s = t0.elapsed().as_secs_f64();
        Ok(eng)
    }

    fn capture_batch_decode_graphs(&mut self) -> Result<()> {
        let toks = [1i64; 4];
        let pos = [8i32; 4];
        let slots: Vec<i32> = (0..self.max_batch as i32).collect();
        let bts = vec![0i32; 4 * MAX_BLOCKS_PER_SEQ];
        self.cuda.htod_sync(self.d_tok_ids, toks.as_ptr() as *const u8, 4 * 8)?;
        self.cuda.htod_sync(self.d_positions, pos.as_ptr() as *const u8, 4 * 4)?;
        self.cuda.htod_sync(self.d_slot_ids, slots.as_ptr() as *const u8, self.max_batch * 4)?;
        self.cuda.htod_sync(self.d_block_tables, bts.as_ptr() as *const u8, bts.len() * 4)?;
        self.cuda.sync()?;

        let mut graphs = Vec::with_capacity(self.max_batch);
        for b in 1..=self.max_batch {
            for _ in 0..2 {
                self.decode_batch_body(b)?;
            }
            self.cuda.sync()?;
            self.cuda.begin_capture()?;
            self.decode_batch_body(b)?;
            let exec = self.cuda.end_capture_and_instantiate()?;
            self.cuda.sync()?;
            graphs.push(exec);
        }
        self.decode_graphs = graphs;
        Ok(())
    }

    pub fn vram_allocated_mib(&self) -> f64 {
        let total = self.weights.p1_bytes
            + self.weights.p2_bytes
            + self.kv_pool.bytes
            + self.dn_pool.bytes
            + self.scratch_bytes
            + self.pf_bytes;
        total as f64 / (1024.0 * 1024.0)
    }

    fn launch_skinny_gemm(&self, m: usize, _n: usize, _k: usize, x: CUdeviceptr, w: Bf12Ptr, y: CUdeviceptr, add_res: bool) -> Result<()> {
        launch_skinny_bf12(&self.cuda, &self.funcs.bf12, m, x, w, y, add_res)
    }

    fn run_stage_paged_prefill(
        &self,
        layers: &[PackedLayerPtrs],
        sub_idxs: &[usize],
        slot_id: usize,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<()> {
        let ws = self.pf_ws;
        let hdr_b = PF_HEADER_ROWS * 2560 * 2;
        let align256 = |b: usize| (b + 255) & !255;
        let m_pad = (seq_len + 3) & !3;

        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut seq_i: i32 = seq_len as i32;
        let mut start_i: i32 = start_pos as i32;
        let mut c8192: i32 = 8192;
        let mut stride_dn: i32 = 12352;
        let mut inter: i32 = 9216;
        let use_init_base: i32 = if start_pos > 0 { 1 } else { 0 };
        let mut snap_zero: i32 = 0;

        for (lyr, &sidx) in layers.iter().zip(sub_idxs.iter()) {
            let mut h_p = ws.p_h;
            let mut nw_p = lyr.in_norm;
            let mut nrm_p = ws.p_nrm;
            self.cuda.launch(
                self.funcs.rmsnorm_2d,
                (seq_len as u32, 1, 1),
                (256, 1, 1),
                &mut [
                    &mut h_p as *mut _ as *mut c_void,
                    &mut nw_p as *mut _ as *mut c_void,
                    &mut nrm_p as *mut _ as *mut c_void,
                    &mut dim as *mut _ as *mut c_void,
                    &mut eps as *mut _ as *mut c_void,
                ],
            )?;

            if lyr.is_linear {
                let p_in_all = ws.p_pool;
                let in_all_b = align256(m_pad * 12352 * 2);
                run_prefill_gemm_i8(
                    &self.cuda, &self.funcs.r10, self.funcs.residual_add,
                    self.prefill_int8, seq_len, 12352, 2560, 1024,
                    ws.p_nrm, lyr.in_all, self.w_unpack_buf, p_in_all, 0, false,
                    ws.p_del, hdr_b, ws.p_pool + in_all_b as u64, PF_POOL_BYTES - in_all_b,
                )?;

                let p_core_out = ws.p_pool + 25_427_968;
                let p_wy_dn_raw = ws.p_pool + 33_816_576;
                let rec_ptr = self.dn_pool.rec_ptr(slot_id, sidx);

                let mut sub_start = 0usize;
                while sub_start < seq_len {
                    let sub_len = (seq_len - sub_start).min(256);
                    let mut sub_len_i: i32 = sub_len as i32;
                    let mut sub_use_init: i32 = if use_init_base != 0 || sub_start > 0 { 1 } else { 0 };
                    let mut pin = p_in_all + (sub_start * 12352 * 2) as u64;
                    let sub_co = p_core_out + (sub_start * 4096 * 2) as u64;
                    let mut cs = self.dn_pool.conv_ptr(slot_id, sidx);
                    let mut cs_snap: CUdeviceptr = 0;
                    let mut cw = lyr.conv_w;
                    let mut cv_out = ws.p_del;

                    if sub_len >= 64 {
                        let t_tiles = ((sub_len + 31) / 32) as u32;
                        self.cuda.launch(
                            self.funcs.conv1d_prefill_2d,
                            (32, t_tiles, 1),
                            (256, 1, 1),
                            &mut [
                                &mut pin as *mut _ as *mut c_void,
                                &mut cs as *mut _ as *mut c_void,
                                &mut cs_snap as *mut _ as *mut c_void,
                                &mut cw as *mut _ as *mut c_void,
                                &mut cv_out as *mut _ as *mut c_void,
                                &mut sub_len_i as *mut _ as *mut c_void,
                                &mut c8192 as *mut _ as *mut c_void,
                                &mut stride_dn as *mut _ as *mut c_void,
                                &mut sub_use_init as *mut _ as *mut c_void,
                                &mut snap_zero as *mut _ as *mut c_void,
                            ],
                        )?;
                        self.cuda.launch(
                            self.funcs.conv1d_update_state,
                            (32, 1, 1),
                            (256, 1, 1),
                            &mut [
                                &mut pin as *mut _ as *mut c_void,
                                &mut cs as *mut _ as *mut c_void,
                                &mut sub_len_i as *mut _ as *mut c_void,
                                &mut c8192 as *mut _ as *mut c_void,
                                &mut stride_dn as *mut _ as *mut c_void,
                                &mut sub_use_init as *mut _ as *mut c_void,
                            ],
                        )?;
                    } else {
                        self.cuda.launch(
                            self.funcs.conv1d_prefill,
                            (32, 1, 1),
                            (256, 1, 1),
                            &mut [
                                &mut pin as *mut _ as *mut c_void,
                                &mut cs as *mut _ as *mut c_void,
                                &mut cs_snap as *mut _ as *mut c_void,
                                &mut cw as *mut _ as *mut c_void,
                                &mut cv_out as *mut _ as *mut c_void,
                                &mut sub_len_i as *mut _ as *mut c_void,
                                &mut c8192 as *mut _ as *mut c_void,
                                &mut stride_dn as *mut _ as *mut c_void,
                                &mut sub_use_init as *mut _ as *mut c_void,
                                &mut snap_zero as *mut _ as *mut c_void,
                            ],
                        )?;
                    }

                    if sub_len >= 192 {
                        run_deltanet_prefill_wy(
                            &self.cuda,
                            &self.funcs.wy,
                            cv_out,
                            pin,
                            lyr.dt_bias,
                            lyr.a_log_f32,
                            lyr.dn_norm,
                            rec_ptr,
                            0,
                            p_wy_dn_raw,
                            ws.p_nrm,
                            sub_co,
                            sub_len,
                            sub_use_init,
                            0,
                            eps,
                        )?;
                    } else {
                        let mut dtb = lyr.dt_bias;
                        let mut alog = lyr.a_log_f32;
                        let mut dnw = lyr.dn_norm;
                        let mut rec = rec_ptr;
                        let mut rec_snap: CUdeviceptr = 0;
                        let mut raw_p = p_wy_dn_raw;
                        let mut co = sub_co;
                        self.cuda.launch(
                            self.funcs.deltanet_prefill,
                            (32, 4, 1),
                            (128, 1, 1),
                            &mut [
                                &mut cv_out as *mut _ as *mut c_void,
                                &mut pin as *mut _ as *mut c_void,
                                &mut dtb as *mut _ as *mut c_void,
                                &mut alog as *mut _ as *mut c_void,
                                &mut rec as *mut _ as *mut c_void,
                                &mut rec_snap as *mut _ as *mut c_void,
                                &mut raw_p as *mut _ as *mut c_void,
                                &mut sub_len_i as *mut _ as *mut c_void,
                                &mut sub_use_init as *mut _ as *mut c_void,
                                &mut snap_zero as *mut _ as *mut c_void,
                            ],
                        )?;
                        self.cuda.launch(
                            self.funcs.deltanet_prefill_norm_gate,
                            (32, sub_len as u32, 1),
                            (128, 1, 1),
                            &mut [
                                &mut raw_p as *mut _ as *mut c_void,
                                &mut pin as *mut _ as *mut c_void,
                                &mut dnw as *mut _ as *mut c_void,
                                &mut co as *mut _ as *mut c_void,
                                &mut eps as *mut _ as *mut c_void,
                            ],
                        )?;
                    }
                    sub_start += sub_len;
                }

                run_prefill_gemm_i8(
                    &self.cuda, &self.funcs.r10, self.funcs.residual_add,
                    self.prefill_int8, seq_len, 2560, 4096, 1024,
                    p_core_out, lyr.out_proj, self.w_unpack_buf, ws.p_del, 0, false,
                    ws.p_nrm, hdr_b, ws.p_pool, 25_427_968,
                )?;
            } else {
                let p_qkv = ws.p_pool;
                let qkv_b = align256(m_pad * 10240 * 2);
                run_prefill_gemm_i8(
                    &self.cuda, &self.funcs.r10, self.funcs.residual_add,
                    self.prefill_int8, seq_len, 10240, 2560, 1024,
                    ws.p_nrm, lyr.qkv, self.w_unpack_buf, p_qkv, 0, false,
                    ws.p_del, hdr_b, ws.p_pool + qkv_b as u64, PF_POOL_BYTES - qkv_b,
                )?;
                let mut qkv_p = p_qkv;
                let mut qnw = lyr.q_norm;
                let mut knw = lyr.k_norm;
                let mut cos_p = self.cos_table;
                let mut sin_p = self.sin_table;
                let mut q_sdpa = ws.p_pool + 20_971_520;
                let p_core_out = ws.p_pool + 29_360_128;
                let mut kvp = self.kv_pool.arena;
                let mut bt_p = self.d_pf_bt;
                let mut fa_i: i32 = sidx as i32;
                self.cuda.launch(
                    self.funcs.qk_rope_paged_prefill,
                    (20, seq_len as u32, 1),
                    (256, 1, 1),
                    &mut [
                        &mut qkv_p as *mut _ as *mut c_void,
                        &mut qnw as *mut _ as *mut c_void,
                        &mut knw as *mut _ as *mut c_void,
                        &mut cos_p as *mut _ as *mut c_void,
                        &mut sin_p as *mut _ as *mut c_void,
                        &mut start_i as *mut _ as *mut c_void,
                        &mut seq_i as *mut _ as *mut c_void,
                        &mut q_sdpa as *mut _ as *mut c_void,
                        &mut kvp as *mut _ as *mut c_void,
                        &mut bt_p as *mut _ as *mut c_void,
                        &mut fa_i as *mut _ as *mut c_void,
                        &mut eps as *mut _ as *mut c_void,
                    ],
                )?;
                let mut co = p_core_out;
                if seq_len >= 64 {
                    let q_tiles = ((seq_len + 31) / 32) as u32;
                    self.cuda.launch(
                        self.funcs.prefill_gqa_paged_wmma_32x16,
                        (16, q_tiles, 1),
                        (128, 1, 1),
                        &mut [
                            &mut q_sdpa as *mut _ as *mut c_void,
                            &mut qkv_p as *mut _ as *mut c_void,
                            &mut kvp as *mut _ as *mut c_void,
                            &mut bt_p as *mut _ as *mut c_void,
                            &mut fa_i as *mut _ as *mut c_void,
                            &mut start_i as *mut _ as *mut c_void,
                            &mut seq_i as *mut _ as *mut c_void,
                            &mut co as *mut _ as *mut c_void,
                        ],
                    )?;
                } else {
                    let q_tiles = ((seq_len + 1) / 2) as u32;
                    self.cuda.launch(
                        self.funcs.prefill_gqa_paged,
                        (4, q_tiles, 1),
                        (256, 1, 1),
                        &mut [
                            &mut q_sdpa as *mut _ as *mut c_void,
                            &mut qkv_p as *mut _ as *mut c_void,
                            &mut kvp as *mut _ as *mut c_void,
                            &mut bt_p as *mut _ as *mut c_void,
                            &mut fa_i as *mut _ as *mut c_void,
                            &mut start_i as *mut _ as *mut c_void,
                            &mut seq_i as *mut _ as *mut c_void,
                            &mut co as *mut _ as *mut c_void,
                        ],
                    )?;
                }
                run_prefill_gemm_i8(
                    &self.cuda, &self.funcs.r10, self.funcs.residual_add,
                    self.prefill_int8, seq_len, 2560, 4096, 1024,
                    p_core_out, lyr.o_proj, self.w_unpack_buf, ws.p_del, 0, false,
                    ws.p_nrm, hdr_b, ws.p_pool, 29_360_128,
                )?;
            }

            let mut del_p = ws.p_del;
            let mut pnw = lyr.post_norm;
            self.cuda.launch(
                self.funcs.add_rmsnorm_2d,
                (seq_len as u32, 1, 1),
                (256, 1, 1),
                &mut [
                    &mut h_p as *mut _ as *mut c_void,
                    &mut del_p as *mut _ as *mut c_void,
                    &mut pnw as *mut _ as *mut c_void,
                    &mut nrm_p as *mut _ as *mut c_void,
                    &mut dim as *mut _ as *mut c_void,
                    &mut eps as *mut _ as *mut c_void,
                ],
            )?;

            if self.prefill_int8 == PrefillInt8Mode::Smooth && seq_len >= 64 && m_pad <= 1024 {
                run_prefill_fused_tiled_mlp_i8(
                    &self.cuda,
                    &self.funcs.r10,
                    seq_len,
                    ws.p_nrm,
                    lyr.gate_up,
                    lyr.down,
                    ws.p_h,
                    ws.p_del,
                    ws.p_pool,
                )?;
            } else {
                let ws_big = ws.p_pool;
                let ws_med = ws.p_pool + 31_457_280;
                run_prefill_gemm_i8(
                    &self.cuda, &self.funcs.r10, self.funcs.residual_add,
                    self.prefill_int8, seq_len, 18432, 2560, 512,
                    ws.p_nrm, lyr.gate_up, self.w_unpack_buf, ws_big, 0, false,
                    ws.p_del, hdr_b, ws.p_pool + 18_874_368, PF_POOL_BYTES - 18_874_368,
                )?;
                let mut gu_p = ws_big;
                let mut act_p = ws_med;
                self.cuda.launch(
                    self.funcs.silu_mul_packed_2d,
                    (36, seq_len as u32, 1),
                    (256, 1, 1),
                    &mut [
                        &mut gu_p as *mut _ as *mut c_void,
                        &mut act_p as *mut _ as *mut c_void,
                        &mut inter as *mut _ as *mut c_void,
                        &mut seq_i as *mut _ as *mut c_void,
                    ],
                )?;
                run_prefill_gemm_i8(
                    &self.cuda, &self.funcs.r10, self.funcs.residual_add,
                    self.prefill_int8, seq_len, 2560, 9216, 512,
                    ws_med, lyr.down, self.w_unpack_buf, ws.p_h, ws.p_del, true,
                    ws.p_pool, 31_457_280, 0, 0,
                )?;
            }
        }
        Ok(())
    }

    /// Phase R11 + R14: 1024-Token Super-Chunk Paged Prefill (BF12 + Smooth INT8) -> returns first_tok!
    pub fn prefill_one_paged(&self, prompt_ids: &[i64], slot_id: usize, blocks: &[i32]) -> Result<i64> {
        self.cuda.htod_async(self.d_pf_bt, blocks.as_ptr() as *const u8, blocks.len() * 4)?;
        let chunk: usize = if self.prefill_int8 == PrefillInt8Mode::Smooth { 1024 } else { 512 };
        let mut cur_pos = 0usize;
        let mut last_seq_len = 0usize;

        while cur_pos < prompt_ids.len() {
            let end = (cur_pos + chunk).min(prompt_ids.len());
            let seg = &prompt_ids[cur_pos..end];
            let seq_len = seg.len();
            self.cuda.htod_async(self.pf_ws.p_ids, seg.as_ptr() as *const u8, seq_len * 8)?;

            launch_embed_prefill_bf12(&self.cuda, &self.funcs.bf12, self.pf_ws.p_ids, self.weights.embed_w, self.pf_ws.p_h, seq_len)?;
            self.run_stage_paged_prefill(&self.weights.p1_layers, &self.p1_sub_idx, slot_id, cur_pos, seq_len)?;
            self.run_stage_paged_prefill(&self.weights.p2_layers, &self.p2_sub_idx, slot_id, cur_pos, seq_len)?;
            cur_pos = end;
            last_seq_len = seq_len;
        }

        // Last token final RMSNorm -> BF12 Fused lm_head_argmax -> first_tok (ZERO d_logits!)
        let last_row = self.pf_ws.p_h + ((last_seq_len - 1) * 2560 * 2) as u64;
        self.cuda.dtod_async(self.d_bridge, last_row, 2560 * 2)?;
        let mut br_p = self.d_bridge;
        let mut fnw = self.weights.final_norm_w;
        let mut nrm = self.d_normed;
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        self.cuda.launch(
            self.funcs.rmsnorm_2d,
            (1, 1, 1),
            (256, 1, 1),
            &mut [
                &mut br_p as *mut _ as *mut c_void,
                &mut fnw as *mut _ as *mut c_void,
                &mut nrm as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;
        launch_lm_head_bf12_argmax(
            &self.cuda,
            &self.funcs.bf12,
            self.funcs.argmax_stage2,
            self.funcs.argmax_batch_stage2,
            1,
            self.d_normed,
            self.weights.lm_head_w,
            self.d_arg_vals,
            self.d_arg_idxs,
            self.d_out_toks,
        )?;
        self.cuda.dtoh_async(self.h_out_toks as *mut u8, self.d_out_toks, 8)?;
        self.cuda.sync()?;
        Ok(unsafe { *self.h_out_toks })
    }

    fn run_stage_batch_decode(
        &self,
        layers: &[PackedLayerPtrs],
        sub_idxs: &[usize],
        b: usize,
        h_buf: CUdeviceptr,
    ) -> Result<()> {
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut c8192: i32 = 8192;
        let mut inter: i32 = 9216;
        let mut b_i: i32 = b as i32;
        let mut max_bps_i: i32 = MAX_BLOCKS_PER_SEQ as i32;
        let mut n_splits: i32 = 4;

        for (lyr, &sidx) in layers.iter().zip(sub_idxs.iter()) {
            let mut h_p = h_buf;
            let mut nw_p = lyr.in_norm;
            let mut nrm_p = self.d_normed;
            self.cuda.launch(
                self.funcs.rmsnorm_2d,
                (b as u32, 1, 1),
                (256, 1, 1),
                &mut [
                    &mut h_p as *mut _ as *mut c_void,
                    &mut nw_p as *mut _ as *mut c_void,
                    &mut nrm_p as *mut _ as *mut c_void,
                    &mut dim as *mut _ as *mut c_void,
                    &mut eps as *mut _ as *mut c_void,
                ],
            )?;

            if lyr.is_linear {
                self.launch_skinny_gemm(b, 12352, 2560, self.d_normed, lyr.in_all, self.d_dn_in, false)?;
                let mut pin = self.d_dn_in;
                let mut cpool = self.dn_pool.conv_arena;
                let mut slts = self.d_slot_ids;
                let mut cw = lyr.conv_w;
                let mut cv_out = self.d_qkv_conv;
                let mut lin_i: i32 = sidx as i32;
                self.cuda.launch(
                    self.funcs.conv1d_batch_decode,
                    (32, b as u32, 1),
                    (256, 1, 1),
                    &mut [
                        &mut pin as *mut _ as *mut c_void,
                        &mut cpool as *mut _ as *mut c_void,
                        &mut slts as *mut _ as *mut c_void,
                        &mut cw as *mut _ as *mut c_void,
                        &mut cv_out as *mut _ as *mut c_void,
                        &mut lin_i as *mut _ as *mut c_void,
                        &mut c8192 as *mut _ as *mut c_void,
                    ],
                )?;

                let mut dtb = lyr.dt_bias;
                let mut alog = lyr.a_log_f32;
                let mut dnw = lyr.dn_norm;
                let mut rpool = self.dn_pool.rec_arena;
                let mut co = self.d_core_out;
                self.cuda.launch(
                    self.funcs.deltanet_batch_decode,
                    (32, b as u32, 1),
                    (256, 1, 1),
                    &mut [
                        &mut cv_out as *mut _ as *mut c_void,
                        &mut pin as *mut _ as *mut c_void,
                        &mut dtb as *mut _ as *mut c_void,
                        &mut alog as *mut _ as *mut c_void,
                        &mut dnw as *mut _ as *mut c_void,
                        &mut rpool as *mut _ as *mut c_void,
                        &mut slts as *mut _ as *mut c_void,
                        &mut co as *mut _ as *mut c_void,
                        &mut lin_i as *mut _ as *mut c_void,
                        &mut eps as *mut _ as *mut c_void,
                    ],
                )?;
                self.launch_skinny_gemm(b, 2560, 4096, self.d_core_out, lyr.out_proj, self.d_delta, false)?;
            } else {
                self.launch_skinny_gemm(b, 10240, 2560, self.d_normed, lyr.qkv, self.d_fa_qkv, false)?;
                let mut qkv_p = self.d_fa_qkv;
                let mut qnw = lyr.q_norm;
                let mut knw = lyr.k_norm;
                let mut cos_p = self.cos_table;
                let mut sin_p = self.sin_table;
                let mut pos_p = self.d_positions;
                let mut qrot = self.d_q_rot;
                let mut kvp = self.kv_pool.arena;
                let mut bts = self.d_block_tables;
                let mut fa_i: i32 = sidx as i32;
                self.cuda.launch(
                    self.funcs.qk_rope_paged_batch_decode,
                    (20, b as u32, 1),
                    (256, 1, 1),
                    &mut [
                        &mut qkv_p as *mut _ as *mut c_void,
                        &mut qnw as *mut _ as *mut c_void,
                        &mut knw as *mut _ as *mut c_void,
                        &mut cos_p as *mut _ as *mut c_void,
                        &mut sin_p as *mut _ as *mut c_void,
                        &mut pos_p as *mut _ as *mut c_void,
                        &mut qrot as *mut _ as *mut c_void,
                        &mut kvp as *mut _ as *mut c_void,
                        &mut bts as *mut _ as *mut c_void,
                        &mut max_bps_i as *mut _ as *mut c_void,
                        &mut fa_i as *mut _ as *mut c_void,
                        &mut eps as *mut _ as *mut c_void,
                    ],
                )?;

                let mut fo = self.d_flash_out;
                let mut fm = self.d_flash_ml;
                self.cuda.launch(
                    self.funcs.paged_gqa_flash_splitk,
                    (16, 4, b as u32),
                    (256, 1, 1),
                    &mut [
                        &mut qrot as *mut _ as *mut c_void,
                        &mut kvp as *mut _ as *mut c_void,
                        &mut bts as *mut _ as *mut c_void,
                        &mut pos_p as *mut _ as *mut c_void,
                        &mut fo as *mut _ as *mut c_void,
                        &mut fm as *mut _ as *mut c_void,
                        &mut max_bps_i as *mut _ as *mut c_void,
                        &mut fa_i as *mut _ as *mut c_void,
                        &mut n_splits as *mut _ as *mut c_void,
                    ],
                )?;
                let mut co = self.d_core_out;
                self.cuda.launch(
                    self.funcs.gqa_verify_m_reduce,
                    (16, b as u32, 1),
                    (256, 1, 1),
                    &mut [
                        &mut fo as *mut _ as *mut c_void,
                        &mut fm as *mut _ as *mut c_void,
                        &mut qkv_p as *mut _ as *mut c_void,
                        &mut co as *mut _ as *mut c_void,
                        &mut n_splits as *mut _ as *mut c_void,
                    ],
                )?;
                self.launch_skinny_gemm(b, 2560, 4096, self.d_core_out, lyr.o_proj, self.d_delta, false)?;
            }

            let mut del_p = self.d_delta;
            let mut pnw = lyr.post_norm;
            self.cuda.launch(
                self.funcs.add_rmsnorm_2d,
                (b as u32, 1, 1),
                (256, 1, 1),
                &mut [
                    &mut h_p as *mut _ as *mut c_void,
                    &mut del_p as *mut _ as *mut c_void,
                    &mut pnw as *mut _ as *mut c_void,
                    &mut nrm_p as *mut _ as *mut c_void,
                    &mut dim as *mut _ as *mut c_void,
                    &mut eps as *mut _ as *mut c_void,
                ],
            )?;

            self.launch_skinny_gemm(b, 18432, 2560, self.d_normed, lyr.gate_up, self.d_mlp_gu, false)?;
            let mut gu_p = self.d_mlp_gu;
            let mut act_p = self.d_mlp_act;
            self.cuda.launch(
                self.funcs.silu_mul_packed_2d,
                (36, b as u32, 1),
                (256, 1, 1),
                &mut [
                    &mut gu_p as *mut _ as *mut c_void,
                    &mut act_p as *mut _ as *mut c_void,
                    &mut inter as *mut _ as *mut c_void,
                    &mut b_i as *mut _ as *mut c_void,
                ],
            )?;
            // Fused Residual Add directly into h_buf!
            self.launch_skinny_gemm(b, 2560, 9216, self.d_mlp_act, lyr.down, h_buf, true)?;
        }
        Ok(())
    }

    fn decode_batch_body(&self, b: usize) -> Result<()> {
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut total_h: i32 = (b * 2560) as i32;

        // PART 1 (Stage 0)
        let mut hid_p = self.d_hidden;
        launch_embed_prefill_bf12(&self.cuda, &self.funcs.bf12, self.d_tok_ids, self.weights.embed_w, self.d_hidden, b)?;
        self.run_stage_batch_decode(&self.weights.p1_layers, &self.p1_sub_idx, b, self.d_hidden)?;

        // ZERO-COPY GPU INTER-STAGE BRIDGE (Part 1 -> Part 2)
        let mut br_p = self.d_bridge;
        let blocks = ((b * 2560 + 255) / 256) as u32;
        self.cuda.launch(
            self.funcs.copy_bf16,
            (blocks, 1, 1),
            (256, 1, 1),
            &mut [
                &mut hid_p as *mut _ as *mut c_void,
                &mut br_p as *mut _ as *mut c_void,
                &mut total_h as *mut _ as *mut c_void,
            ],
        )?;

        // PART 2 (Stage 1)
        self.run_stage_batch_decode(&self.weights.p2_layers, &self.p2_sub_idx, b, self.d_bridge)?;

        let mut fnw = self.weights.final_norm_w;
        let mut nrm = self.d_normed;
        self.cuda.launch(
            self.funcs.rmsnorm_2d,
            (b as u32, 1, 1),
            (256, 1, 1),
            &mut [
                &mut br_p as *mut _ as *mut c_void,
                &mut fnw as *mut _ as *mut c_void,
                &mut nrm as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;
        // Phase R14: Fused Batched BF12 lm_head + Online Argmax for B=1..4 (ZERO d_logits, ZERO cuBLAS!)
        launch_lm_head_bf12_argmax(
            &self.cuda,
            &self.funcs.bf12,
            self.funcs.argmax_stage2,
            self.funcs.argmax_batch_stage2,
            b,
            self.d_normed,
            self.weights.lm_head_w,
            self.d_arg_vals,
            self.d_arg_idxs,
            self.d_out_toks,
        )?;
        // Phase R9: Capture D2H directly inside decode_graphs[b-1] (0 extra host launch!)
        self.cuda.dtoh_async(self.h_out_toks as *mut u8, self.d_out_toks, b * 8)?;
        Ok(())
    }

    /// Execute 1 Batched Decode Step for B = 1..4 concurrent requests via 1 PINNED HtoD + 1 CUDA GRAPH LAUNCH!
    pub fn decode_step_batch(
        &self,
        tok_ids: &[i64],
        positions: &[i32],
        slot_ids: &[i32],
        block_tables: &[&[i32]],
    ) -> Result<[i64; 4]> {
        let b = tok_ids.len();
        unsafe {
            std::ptr::copy_nonoverlapping(tok_ids.as_ptr() as *const u8, self.h_meta_pinned, b * 8);
            std::ptr::copy_nonoverlapping(positions.as_ptr() as *const u8, self.h_meta_pinned.add(self.meta_off_pos), b * 4);
            std::ptr::copy_nonoverlapping(slot_ids.as_ptr() as *const u8, self.h_meta_pinned.add(self.meta_off_slt), b * 4);
            for (i, bt) in block_tables.iter().enumerate() {
                let off = self.meta_off_bt + i * MAX_BLOCKS_PER_SEQ * 4;
                std::ptr::copy_nonoverlapping(bt.as_ptr() as *const u8, self.h_meta_pinned.add(off), bt.len() * 4);
            }
        }
        // ONE single pinned HtoD (~2 us, eliminates 7x pageable WDDM staging copies!)
        let copy_end = self.meta_off_bt + b * MAX_BLOCKS_PER_SEQ * 4;
        self.cuda.htod_async(self.d_tok_ids, self.h_meta_pinned, copy_end.min(self.meta_bytes))?;

        // 1 SINGLE CUDA Graph launch (includes ~310 kernels + D2H!)
        self.cuda.launch_graph(self.decode_graphs[b - 1])?;
        self.cuda.sync()?;

        let mut out = [0i64; 4];
        unsafe {
            for i in 0..b {
                out[i] = *self.h_out_toks.add(i);
            }
        }
        Ok(out)
    }

    /// Profile helper: runs 64 Batched Decode Steps at B = max_batch (for nsys 10 kHz capture!)
    pub fn profile_batch_decode_64(&mut self, prompt_ids: &[i64]) -> Result<f64> {
        let b = self.max_batch;
        let mut slots = Vec::with_capacity(b);
        let mut bts: Vec<Vec<i32>> = Vec::with_capacity(b);
        let mut cur_toks = Vec::with_capacity(b);
        let mut positions = vec![prompt_ids.len() as i32; b];
        let total_blks = KvBlockPool::blocks_needed(prompt_ids.len() + 66);

        for _ in 0..b {
            let s = self.dn_pool.acquire().unwrap();
            let blks = self.kv_pool.alloc_blocks(total_blks).unwrap();
            let ft = self.prefill_one_paged(prompt_ids, s, &blks)?;
            slots.push(s as i32);
            bts.push(blks);
            cur_toks.push(ft);
        }
        self.cuda.sync()?;

        let t0 = Instant::now();
        for _ in 0..64 {
            let bt_refs: Vec<&[i32]> = bts.iter().map(|v| v.as_slice()).collect();
            let next_toks = self.decode_step_batch(&cur_toks, &positions, &slots, &bt_refs)?;
            for i in 0..b {
                cur_toks[i] = next_toks[i];
                positions[i] += 1;
            }
        }
        self.cuda.sync()?;
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

        for (s, blks) in slots.into_iter().zip(bts.into_iter()) {
            self.dn_pool.release(s as usize);
            self.kv_pool.free_blocks(&blks);
        }
        Ok(elapsed_ms)
    }
}

/// Non-blocking handle held by Axum `AppState` to submit concurrent requests into the GPU scheduler thread!
pub struct BatchClient {
    pub tok: Arc<QwenTokenizer>,
    pub req_tx: std_mpsc::Sender<BatchSubmitReq>,
    pub status: Arc<Mutex<BatchStatusSnapshot>>,
}

impl BatchClient {
    pub fn start(
        model_dir: &Path,
        split_layer: usize,
        max_seq: usize,
        max_batch: usize,
        force_max_seq: bool,
        prefill_int8: PrefillInt8Mode,
    ) -> Result<Arc<Self>> {
        let eng = BatchGpuEngine::load(model_dir, split_layer, max_seq, max_batch, force_max_seq, prefill_int8)?;
        let (free_b, total_b) = eng.cuda.mem_info().unwrap_or((0, 0));
        let tok = Arc::new(QwenTokenizer::load(model_dir, eng.cfg.eos_token_id)?);
        let init_status = BatchStatusSnapshot {
            split_layer: eng.split_layer,
            max_seq: eng.max_seq,
            max_batch: eng.max_batch,
            active_batch: 0,
            queued_requests: 0,
            kv_blocks_used: 0,
            kv_blocks_total: eng.kv_pool.total_blocks,
            kv_block_size: KV_BLOCK_SIZE,
            load_time_s: eng.load_time_s,
            vram_allocated_mib: eng.vram_allocated_mib(),
            vram_free_mib: free_b as f64 / 1048576.0,
            vram_total_mib: total_b as f64 / 1048576.0,
        };
        let status = Arc::new(Mutex::new(init_status));
        let status_bg = status.clone();
        let (req_tx, req_rx) = std_mpsc::channel::<BatchSubmitReq>();

        thread::Builder::new()
            .name("gpu-batch-scheduler".into())
            .spawn(move || {
                if let Err(e) = run_scheduler_loop(eng, req_rx, status_bg) {
                    eprintln!("[BatchScheduler] fatal error: {e:#}");
                }
            })?;

        Ok(Arc::new(Self { tok, req_tx, status }))
    }
}

fn update_status(
    eng: &BatchGpuEngine,
    status: &Arc<Mutex<BatchStatusSnapshot>>,
    active_len: usize,
    queued_len: usize,
) {
    let (free_b, total_b) = eng.cuda.mem_info().unwrap_or((0, 0));
    if let Ok(mut st) = status.lock() {
        st.active_batch = active_len;
        st.queued_requests = queued_len;
        st.kv_blocks_used = eng.kv_pool.used_count();
        st.vram_free_mib = free_b as f64 / 1048576.0;
        st.vram_total_mib = total_b as f64 / 1048576.0;
    }
}

fn run_scheduler_loop(
    mut eng: BatchGpuEngine,
    req_rx: std_mpsc::Receiver<BatchSubmitReq>,
    status: Arc<Mutex<BatchStatusSnapshot>>,
) -> Result<()> {
    eng.cuda.bind_context()?;
    let mut waiting: VecDeque<BatchSubmitReq> = VecDeque::new();
    let mut active: Vec<ActiveSeq> = Vec::with_capacity(eng.max_batch);

    loop {
        // 1. Collect newly arrived requests (blocking only when completely idle!)
        if active.is_empty() && waiting.is_empty() {
            match req_rx.recv() {
                Ok(r) => waiting.push_back(r),
                Err(_) => break, // channel closed -> clean exit
            }
        }
        while let Ok(r) = req_rx.try_recv() {
            waiting.push_back(r);
        }

        // 2. Iteration-level Admission: admit waiting requests whenever a DeltaNet slot + KV blocks are available!
        while active.len() < eng.max_batch && !waiting.is_empty() {
            let needed_toks = waiting.front().unwrap().prompt_ids.len().min(eng.max_seq - 2);
            let init_blks = KvBlockPool::blocks_needed(needed_toks + 1);
            if eng.kv_pool.free_count() < init_blks {
                break; // wait for an active request to finish & free KV blocks!
            }
            let Some(slot_id) = eng.dn_pool.acquire() else { break; };
            let req = waiting.pop_front().unwrap();
            let mut pids = req.prompt_ids;
            let mut max_new = req.max_new_tokens.max(1);
            if pids.len() + 2 > eng.max_seq {
                let keep = eng.max_seq - 2;
                pids = pids[pids.len() - keep..].to_vec();
            }
            let prompt_len = pids.len();
            if prompt_len + max_new > eng.max_seq {
                max_new = eng.max_seq.saturating_sub(prompt_len).max(1);
            }
            let blks = eng.kv_pool.alloc_blocks(KvBlockPool::blocks_needed(prompt_len)).unwrap();

            let first_tok = match eng.prefill_one_paged(&pids, slot_id, &blks) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[BatchScheduler] prefill error: {e:#}");
                    eng.dn_pool.release(slot_id);
                    eng.kv_pool.free_blocks(&blks);
                    continue;
                }
            };

            let is_eos = eng.tok.is_eos(first_tok) && !req.ignore_eos;
            let done_now = is_eos || max_new <= 1;
            let keep = req.tx.send(StreamToken {
                token_id: first_tok,
                text: if is_eos { String::new() } else { eng.tok.decode_single(first_tok) },
                prompt_tokens: prompt_len,
                cached_tokens: 0,
                cache_tier: "paged_kv",
                cache_restore_ms: 0.0,
                completion_tokens: 1,
                finish_reason: if done_now {
                    Some(if is_eos { "stop" } else { "length" })
                } else {
                    None
                },
            }).is_ok();

            if done_now || !keep {
                eng.dn_pool.release(slot_id);
                eng.kv_pool.free_blocks(&blks);
            } else {
                active.push(ActiveSeq {
                    slot_id,
                    blocks: blks,
                    prompt_len,
                    pos: prompt_len,
                    cur_tok: first_tok,
                    emitted: 1,
                    max_new_tokens: max_new,
                    ignore_eos: req.ignore_eos,
                    tx: req.tx,
                });
            }
            // Pick up any requests that arrived during prefill so they can join BEFORE the next decode!
            while let Ok(r) = req_rx.try_recv() {
                waiting.push_back(r);
            }
        }

        if active.is_empty() {
            update_status(&eng, &status, 0, waiting.len());
            continue;
        }

        // 3. Allocate 1 KV block for any active sequence crossing a 16-token page boundary
        let mut evicted_oom = Vec::new();
        for (idx, seq) in active.iter_mut().enumerate() {
            let req_blks = KvBlockPool::blocks_needed(seq.pos + 1);
            if seq.blocks.len() < req_blks {
                if let Some(nb) = eng.kv_pool.alloc_one() {
                    seq.blocks.push(nb);
                } else {
                    evicted_oom.push(idx);
                }
            }
        }
        for idx in evicted_oom.into_iter().rev() {
            let seq = active.swap_remove(idx);
            let _ = seq.tx.send(StreamToken {
                token_id: 0,
                text: String::new(),
                prompt_tokens: seq.prompt_len,
                cached_tokens: 0,
                cache_tier: "paged_kv",
                cache_restore_ms: 0.0,
                completion_tokens: seq.emitted,
                finish_reason: Some("length"),
            });
            eng.dn_pool.release(seq.slot_id);
            eng.kv_pool.free_blocks(&seq.blocks);
        }
        if active.is_empty() {
            update_status(&eng, &status, 0, waiting.len());
            continue;
        }

        // 4. Execute ONE Batched Decode Step for ALL B active requests simultaneously!
        let b = active.len();
        let mut tok_ids = [0i64; 4];
        let mut positions = [0i32; 4];
        let mut slot_ids = [0i32; 4];
        for (i, seq) in active.iter().enumerate() {
            tok_ids[i] = seq.cur_tok;
            positions[i] = seq.pos as i32;
            slot_ids[i] = seq.slot_id as i32;
        }
        let bt_refs: Vec<&[i32]> = active.iter().map(|s| s.blocks.as_slice()).collect();
        let next_toks = eng.decode_step_batch(&tok_ids[..b], &positions[..b], &slot_ids[..b], &bt_refs)?;

        // 5. Stream newly generated tokens & immediately evict finished/aborted requests!
        let mut to_remove = Vec::new();
        for (i, seq) in active.iter_mut().enumerate() {
            let tid = next_toks[i];
            seq.pos += 1;
            seq.emitted += 1;
            seq.cur_tok = tid;

            let is_eos = eng.tok.is_eos(tid) && !seq.ignore_eos;
            let hit_len = seq.emitted >= seq.max_new_tokens || seq.pos >= eng.max_seq;
            let finish = if is_eos {
                Some("stop")
            } else if hit_len {
                Some("length")
            } else {
                None
            };
            let keep = seq.tx.send(StreamToken {
                token_id: tid,
                text: if is_eos { String::new() } else { eng.tok.decode_single(tid) },
                prompt_tokens: seq.prompt_len,
                cached_tokens: 0,
                cache_tier: "paged_kv",
                cache_restore_ms: 0.0,
                completion_tokens: seq.emitted,
                finish_reason: finish,
            }).is_ok();

            if finish.is_some() || !keep {
                to_remove.push(i);
            }
        }
        for idx in to_remove.into_iter().rev() {
            let seq = active.swap_remove(idx);
            eng.dn_pool.release(seq.slot_id);
            eng.kv_pool.free_blocks(&seq.blocks);
        }

        update_status(&eng, &status, active.len(), waiting.len());
    }
    Ok(())
}
