//! Pure Rust 2-Part Split GPU Inference Engine for Qwen3.5-4B.
//! Stage 0 (Part 1: Embed + L0..split-1) -> Zero-Copy GPU Bridge -> Stage 1 (Part 2: Lsplit..31 + Norm + Tied LM Head)
//! Zero PyTorch, Zero CuPy, Single-Arena VRAM, CUDA Graph Decode + Overlapped Pinned Host Double-Buffering + Prefix Cache.

use anyhow::Result;
use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::path::Path;
use std::time::Instant;

use crate::config::TextConfig;
use crate::cuda_ffi::{CUdeviceptr, CUevent, CUfunction, CUgraphExec, CudaRuntime};
use crate::loader::{load_split_weights, LoadedSplitWeights, MtpWeights, PackedLayerPtrs};
use crate::tokenizer::QwenTokenizer;

pub(crate) const PTX_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/qwen35.ptx"));

struct KernelFuncs {
    rmsnorm_2d: CUfunction,
    add_rmsnorm_2d: CUfunction,
    silu_mul_packed_2d: CUfunction,
    residual_add: CUfunction,
    copy_bf16: CUfunction,
    conv1d_update: CUfunction,
    conv1d_prefill: CUfunction,
    deltanet_decode_packed: CUfunction,
    deltanet_prefill: CUfunction,
    deltanet_prefill_norm_gate: CUfunction,
    qk_rope_kv_write_packed: CUfunction,
    qk_rope_kv_prefill: CUfunction,
    prefill_gqa_causal_gated: CUfunction,
    gqa_flash_splitk: CUfunction,
    gqa_flash_reduce: CUfunction,
    embed_single: CUfunction,
    embed_prefill: CUfunction,
    argmax_stage1: CUfunction,
    argmax_stage2: CUfunction,
    inc_pos: CUfunction,
    copy_i64: CUfunction,
    // Phase R3: MTP + Parallel Verify
    mtp_pre_fc_concat: CUfunction,
    conv1d_verify_m: CUfunction,
    deltanet_verify_m: CUfunction,
    deltanet_commit_partial: CUfunction,
    qk_rope_kv_verify_m: CUfunction,
    gqa_verify_m_splitk: CUfunction,
    gqa_verify_m_reduce: CUfunction,
    argmax_batch_stage1: CUfunction,
    argmax_batch_stage2: CUfunction,
    // Phase R6: Custom Warp-Vectorized Skinny BF16 GEMM (M=1..4) + Fused Residual Add
    skinny_gemm_m1: CUfunction,
    skinny_gemm_m2: CUfunction,
    skinny_gemm_m3: CUfunction,
    skinny_gemm_m4: CUfunction,
    // Phase R8: Chunked WY-Form Gated DeltaNet Prefill (Ampere Tensor Cores)
    wy: WyKernelFuncs,
    // Phase R9: 2D Parallel Conv1D Prefill + WMMA BF16 Tensor Core FlashAttention-2
    conv1d_prefill_2d: CUfunction,
    conv1d_update_state: CUfunction,
    prefill_gqa_wmma: CUfunction,
}

#[derive(Clone, Copy)]
pub(crate) struct WyKernelFuncs {
    pub cumsum_g: CUfunction,
    pub prep_qk: CUfunction,
    pub solve_2x32: CUfunction,
    pub prep_vb_scale: CUfunction,
    pub state_wmma: CUfunction,
    pub norm_gate_inplace: CUfunction,
}

impl WyKernelFuncs {
    pub fn load(cuda: &CudaRuntime) -> Result<Self> {
        Ok(Self {
            cumsum_g: cuda.get_func("deltanet_wy_cumsum_g_f32")?,
            prep_qk: cuda.get_func("deltanet_wy_prep_qk_bf16")?,
            solve_2x32: cuda.get_func("deltanet_wy_solve_2x32_bf16")?,
            prep_vb_scale: cuda.get_func("deltanet_wy_prep_vb_and_scale_qk_bf16")?,
            state_wmma: cuda.get_func("deltanet_wy_state_wmma_bf16")?,
            norm_gate_inplace: cuda.get_func("deltanet_prefill_norm_gate_inplace_bf16")?,
        })
    }
}

/// Phase R8 v2: FlashLinearAttention-Style Chunked WY Gated DeltaNet Prefill.
/// - TILE_CHUNKS = 4 (256 tok/tile -> batch = 128 Ampere BF16 Tensor Core GEMMs)
/// - 2x32 warp-shuffle block triangular inversion (0 __syncthreads())
/// - Single-kernel WMMA BF16 Tensor Core inter-chunk state + output (0 intermediate S DRAM!)
/// - Reuses existing `dn_raw` (8.0 MiB) + `ws_big` tail (5.5 MiB) -> +0.00 MiB VRAM!
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_deltanet_prefill_wy(
    cuda: &CudaRuntime,
    wy: &WyKernelFuncs,
    qkv_conv: CUdeviceptr,
    packed_in: CUdeviceptr,
    dt_bias: CUdeviceptr,
    a_log: CUdeviceptr,
    dn_norm: CUdeviceptr,
    rec_state: CUdeviceptr,
    rec_snap_out: CUdeviceptr,
    dn_raw: CUdeviceptr,
    ws_big: CUdeviceptr,
    core_out: CUdeviceptr,
    seq_len: usize,
    use_init_state: i32,
    snap_t: i32,
    eps: f32,
) -> Result<()> {
    let mut chunks: Vec<(usize, usize)> = Vec::with_capacity(10);
    let mut push_range = |mut s: usize, end: usize| {
        while s < end {
            let l = (end - s).min(64);
            chunks.push((s, l));
            s += l;
        }
    };
    if snap_t > 0 && (snap_t as usize) < seq_len {
        push_range(0, snap_t as usize);
        push_range(snap_t as usize, seq_len);
    } else {
        push_range(0, seq_len);
    }

    // Zero-extra-VRAM BF16 buffer map for B_c <= 4 chunks (256 tokens/tile):
    //   dn_raw (8.00 MiB): buf_q (2M) | buf_k (2M) | buf_w (2M) | buf_u (2M)
    //   ws_big + 12.50 MiB (5.50 MiB free): buf_t (1M) | buf_a (1M) | buf_tmp (2M) | buf_g/beta/g_last (65 KB)
    let buf_q = dn_raw;
    let buf_k = dn_raw + 2_097_152;
    let buf_w = dn_raw + 4_194_304;
    let buf_u = dn_raw + 6_291_456;

    let tail = ws_big + 13_107_200;
    let buf_t = tail;
    let buf_a = tail + 1_048_576;
    let buf_tmp = tail + 2_097_152;
    let buf_g = tail + 4_194_304;
    let buf_beta = buf_g + 32_768;
    let buf_g_last = buf_beta + 32_768;

    for (tile_idx, tile) in chunks.chunks(4).enumerate() {
        let b_c = tile.len();
        let get_c = |idx: usize| -> (i32, i32) {
            if idx < b_c { (tile[idx].0 as i32, tile[idx].1 as i32) } else { (0, 0) }
        };
        let (mut c0s, mut c0l) = get_c(0);
        let (mut c1s, mut c1l) = get_c(1);
        let (mut c2s, mut c2l) = get_c(2);
        let (mut c3s, mut c3l) = get_c(3);
        let batch_all = 32 * b_c;

        let mut p_conv = qkv_conv;
        let mut p_pin = packed_in;
        let mut p_dtb = dt_bias;
        let mut p_alog = a_log;
        let mut p_q = buf_q;
        let mut p_k = buf_k;
        let mut p_w = buf_w;
        let mut p_u = buf_u;
        let mut p_t = buf_t;
        let mut p_a = buf_a;
        let mut p_tmp = buf_tmp;
        let mut p_g = buf_g;
        let mut p_beta = buf_beta;
        let mut p_gl = buf_g_last;

        // 1. Cumsum log-decay G, beta, g_last
        cuda.launch(
            wy.cumsum_g,
            (32, b_c as u32, 1),
            (64, 1, 1),
            &mut [
                &mut p_pin as *mut _ as *mut c_void,
                &mut p_dtb as *mut _ as *mut c_void,
                &mut p_alog as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_beta as *mut _ as *mut c_void,
                &mut p_gl as *mut _ as *mut c_void,
                &mut c0s as *mut _ as *mut c_void, &mut c0l as *mut _ as *mut c_void,
                &mut c1s as *mut _ as *mut c_void, &mut c1l as *mut _ as *mut c_void,
                &mut c2s as *mut _ as *mut c_void, &mut c2l as *mut _ as *mut c_void,
                &mut c3s as *mut _ as *mut c_void, &mut c3l as *mut _ as *mut c_void,
            ],
        )?;

        // 2. Parallel Q_norm, K_norm, Kb = beta * exp(G) * K (8,192 blocks!)
        cuda.launch(
            wy.prep_qk,
            (32, 64, b_c as u32),
            (128, 1, 1),
            &mut [
                &mut p_conv as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_beta as *mut _ as *mut c_void,
                &mut p_q as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_tmp as *mut _ as *mut c_void,
                &mut c0s as *mut _ as *mut c_void, &mut c0l as *mut _ as *mut c_void,
                &mut c1s as *mut _ as *mut c_void, &mut c1l as *mut _ as *mut c_void,
                &mut c2s as *mut _ as *mut c_void, &mut c2l as *mut _ as *mut c_void,
                &mut c3s as *mut _ as *mut c_void, &mut c3l as *mut _ as *mut c_void,
            ],
        )?;

        // 3 & 4. Ampere BF16 Tensor Core Batched GEMMs (batch=128!): M_raw = K@K^T, A_raw = Q@K^T
        cuda.gemm_bf16_strided_batched(64, 64, 128, false, true, 1.0, buf_k, 8192, buf_k, 8192, 0.0, buf_t, 4096, batch_all)?;
        cuda.gemm_bf16_strided_batched(64, 64, 128, false, true, 1.0, buf_q, 8192, buf_k, 8192, 0.0, buf_a, 4096, batch_all)?;

        // 5. Fast 2x32 Warp-Shuffle Block Triangular Inversion T = (I+M)^-1 + Mask A_causal
        cuda.launch(
            wy.solve_2x32,
            (32, b_c as u32, 1),
            (64, 1, 1),
            &mut [
                &mut p_t as *mut _ as *mut c_void,
                &mut p_a as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_beta as *mut _ as *mut c_void,
                &mut c0l as *mut _ as *mut c_void,
                &mut c1l as *mut _ as *mut c_void,
                &mut c2l as *mut _ as *mut c_void,
                &mut c3l as *mut _ as *mut c_void,
            ],
        )?;

        // 6. Ampere BF16 Tensor Core Batched GEMM: W = T @ Kb
        cuda.gemm_bf16_strided_batched(64, 128, 64, false, false, 1.0, buf_t, 4096, buf_tmp, 8192, 0.0, buf_w, 8192, batch_all)?;

        // 7. Fused Parallel Vb = beta * V + Scale Q *= exp(G_i) & K *= exp(G_last - G_i)
        cuda.launch(
            wy.prep_vb_scale,
            (32, 64, b_c as u32),
            (128, 1, 1),
            &mut [
                &mut p_conv as *mut _ as *mut c_void,
                &mut p_g as *mut _ as *mut c_void,
                &mut p_beta as *mut _ as *mut c_void,
                &mut p_gl as *mut _ as *mut c_void,
                &mut p_q as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_tmp as *mut _ as *mut c_void,
                &mut c0s as *mut _ as *mut c_void, &mut c0l as *mut _ as *mut c_void,
                &mut c1s as *mut _ as *mut c_void, &mut c1l as *mut _ as *mut c_void,
                &mut c2s as *mut _ as *mut c_void, &mut c2l as *mut _ as *mut c_void,
                &mut c3s as *mut _ as *mut c_void, &mut c3l as *mut _ as *mut c_void,
            ],
        )?;

        // 8. Ampere BF16 Tensor Core Batched GEMM: U = T @ Vb
        cuda.gemm_bf16_strided_batched(64, 128, 64, false, false, 1.0, buf_t, 4096, buf_tmp, 8192, 0.0, buf_u, 8192, batch_all)?;

        // 9. SINGLE-KERNEL Fused WMMA BF16 Tensor Core State + Output (0 intermediate S DRAM!)
        let mut p_rec = rec_state;
        let mut p_snap = rec_snap_out;
        let mut p_co = core_out;
        let mut i_bc = b_c as i32;
        let mut i_init = if use_init_state != 0 || tile_idx > 0 { 1i32 } else { 0i32 };
        let mut i_snap = snap_t;
        cuda.launch(
            wy.state_wmma,
            (32, 8, 1),
            (128, 1, 1),
            &mut [
                &mut p_q as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
                &mut p_w as *mut _ as *mut c_void,
                &mut p_u as *mut _ as *mut c_void,
                &mut p_a as *mut _ as *mut c_void,
                &mut p_gl as *mut _ as *mut c_void,
                &mut p_rec as *mut _ as *mut c_void,
                &mut p_snap as *mut _ as *mut c_void,
                &mut p_co as *mut _ as *mut c_void,
                &mut i_bc as *mut _ as *mut c_void,
                &mut i_init as *mut _ as *mut c_void,
                &mut i_snap as *mut _ as *mut c_void,
                &mut c0s as *mut _ as *mut c_void, &mut c0l as *mut _ as *mut c_void,
                &mut c1s as *mut _ as *mut c_void, &mut c1l as *mut _ as *mut c_void,
                &mut c2s as *mut _ as *mut c_void, &mut c2l as *mut _ as *mut c_void,
                &mut c3s as *mut _ as *mut c_void, &mut c3l as *mut _ as *mut c_void,
            ],
        )?;
    }

    // 10. Single Fused In-Place RMSNorm + SiLU(z) gate across all seq_len tokens
    let mut p_co = core_out;
    let mut p_pin = packed_in;
    let mut p_dnw = dn_norm;
    let mut eps_v = eps;
    cuda.launch(
        wy.norm_gate_inplace,
        (32, seq_len as u32, 1),
        (128, 1, 1),
        &mut [
            &mut p_co as *mut _ as *mut c_void,
            &mut p_pin as *mut _ as *mut c_void,
            &mut p_dnw as *mut _ as *mut c_void,
            &mut eps_v as *mut _ as *mut c_void,
        ],
    )?;
    Ok(())
}

impl KernelFuncs {
    fn load(cuda: &CudaRuntime) -> Result<Self> {
        Ok(Self {
            rmsnorm_2d: cuda.get_func("qwen35_rmsnorm_2d_bf16")?,
            add_rmsnorm_2d: cuda.get_func("qwen35_add_rmsnorm_2d_bf16")?,
            silu_mul_packed_2d: cuda.get_func("silu_mul_packed_2d_bf16")?,
            residual_add: cuda.get_func("residual_add_bf16")?,
            copy_bf16: cuda.get_func("copy_bf16")?,
            conv1d_update: cuda.get_func("causal_conv1d_update_k4_bf16")?,
            conv1d_prefill: cuda.get_func("causal_conv1d_prefill_k4_bf16")?,
            deltanet_decode_packed: cuda.get_func("gated_deltanet_decode_packed_bf16")?,
            deltanet_prefill: cuda.get_func("gated_deltanet_prefill_bf16")?,
            deltanet_prefill_norm_gate: cuda.get_func("deltanet_prefill_norm_gate_bf16")?,
            qk_rope_kv_write_packed: cuda.get_func("full_attn_qk_rope_kv_write_packed_bf16")?,
            qk_rope_kv_prefill: cuda.get_func("full_attn_qk_rope_kv_prefill_bf16")?,
            prefill_gqa_causal_gated: cuda.get_func("prefill_gqa_causal_gated_bf16")?,
            gqa_flash_splitk: cuda.get_func("gqa_decode_flash_splitk_bf16")?,
            gqa_flash_reduce: cuda.get_func("gqa_decode_flash_reduce_bf16")?,
            embed_single: cuda.get_func("embed_lookup_single_bf16")?,
            embed_prefill: cuda.get_func("embed_lookup_prefill_bf16")?,
            argmax_stage1: cuda.get_func("argmax_stage1_bf16")?,
            argmax_stage2: cuda.get_func("argmax_stage2_bf16")?,
            inc_pos: cuda.get_func("inc_pos_i32")?,
            copy_i64: cuda.get_func("copy_i64")?,
            mtp_pre_fc_concat: cuda.get_func("mtp_pre_fc_concat_bf16")?,
            conv1d_verify_m: cuda.get_func("causal_conv1d_verify_m_bf16")?,
            deltanet_verify_m: cuda.get_func("gated_deltanet_verify_m_bf16")?,
            deltanet_commit_partial: cuda.get_func("deltanet_commit_partial_bf16")?,
            qk_rope_kv_verify_m: cuda.get_func("full_attn_qk_rope_kv_verify_m_bf16")?,
            gqa_verify_m_splitk: cuda.get_func("gqa_verify_m_flash_splitk_bf16")?,
            gqa_verify_m_reduce: cuda.get_func("gqa_verify_m_flash_reduce_bf16")?,
            argmax_batch_stage1: cuda.get_func("argmax_batch_stage1_bf16")?,
            argmax_batch_stage2: cuda.get_func("argmax_batch_stage2_bf16")?,
            skinny_gemm_m1: cuda.get_func("skinny_gemm_m1_2wpr_bf16")?,
            skinny_gemm_m2: cuda.get_func("skinny_gemm_m2_1wpr_bf16")?,
            skinny_gemm_m3: cuda.get_func("skinny_gemm_m3_1wpr_bf16")?,
            skinny_gemm_m4: cuda.get_func("skinny_gemm_m4_1wpr_bf16")?,
            wy: WyKernelFuncs::load(cuda)?,
            conv1d_prefill_2d: cuda.get_func("causal_conv1d_prefill_2d_bf16")?,
            conv1d_update_state: cuda.get_func("causal_conv1d_update_state_bf16")?,
            prefill_gqa_wmma: cuda.get_func("prefill_gqa_wmma_causal_gated_bf16")?,
        })
    }
}

#[derive(Clone, Copy)]
struct LayerCachePtrs {
    is_linear: bool,
    conv_state: CUdeviceptr, // [8192, 4] bf16
    rec_state: CUdeviceptr,  // [32, 128, 128] f32
    k_cache: CUdeviceptr,    // [4, max_seq, 256] bf16
    v_cache: CUdeviceptr,    // [4, max_seq, 256] bf16
}

struct StageCache {
    arena: CUdeviceptr,
    bytes: usize,
    layers: Vec<LayerCachePtrs>,
}

struct PinnedBuf {
    ptr: *mut u8,
    cap: usize,
    len: usize,
    free_fn: unsafe extern "C" fn(*mut c_void) -> i32,
}

unsafe impl Send for PinnedBuf {}

impl PinnedBuf {
    fn alloc(cuda: &CudaRuntime, cap: usize) -> Result<Self> {
        let alloc_cap = cap.max(16);
        let ptr = cuda.alloc_pinned(alloc_cap)?;
        Ok(Self {
            ptr,
            cap: alloc_cap,
            len: 0,
            free_fn: cuda.free_pinned_fn(),
        })
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                let _ = (self.free_fn)(self.ptr as *mut c_void);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

struct PinnedSnapshot {
    seq_len: usize,
    has_mtp: bool,
    data: Vec<u8>,     // Normal Host RAM (0 MiB GPU WDDM overhead!)
    last_h_off: usize, // byte offset of [2560] bf16 post-norm hidden state inside `data`
}

impl StageCache {
    fn alloc(cuda: &CudaRuntime, layers: &[PackedLayerPtrs], max_seq: usize) -> Result<Self> {
        let align = |b: usize| (b + 255) & !255;
        let conv_sz = align(8192 * 4 * 2);
        let rec_sz = align(32 * 128 * 128 * 4);
        let kv_sz = align(4 * max_seq * 256 * 2);

        let mut total = 0usize;
        for l in layers {
            if l.is_linear {
                total += conv_sz + rec_sz;
            } else {
                total += kv_sz * 2;
            }
        }
        let arena = cuda.alloc(total)?;
        cuda.memset_zero_async(arena, total)?;
        cuda.sync()?;

        let mut off = 0usize;
        let mut out = Vec::with_capacity(layers.len());
        for l in layers {
            if l.is_linear {
                let conv_state = arena + off as u64;
                off += conv_sz;
                let rec_state = arena + off as u64;
                off += rec_sz;
                out.push(LayerCachePtrs { is_linear: true, conv_state, rec_state, k_cache: 0, v_cache: 0 });
            } else {
                let k_cache = arena + off as u64;
                off += kv_sz;
                let v_cache = arena + off as u64;
                off += kv_sz;
                out.push(LayerCachePtrs { is_linear: false, conv_state: 0, rec_state: 0, k_cache, v_cache });
            }
        }
        Ok(Self { arena, bytes: total, layers: out })
    }

    fn calc_snapshot_bytes(&self, seq_len: usize) -> usize {
        let conv_bytes = 8192 * 4 * 2;
        let rec_bytes = 32 * 128 * 128 * 4;
        let kv_layer_bytes = 4 * seq_len * 256 * 2 * 2;
        self.layers
            .iter()
            .map(|l| if l.is_linear { conv_bytes + rec_bytes } else { kv_layer_bytes })
            .sum()
    }

    fn enqueue_dtoh_from_rb(
        &self,
        cuda: &CudaRuntime,
        rbs: Option<&[LinearRollbackPtrs]>,
        dst: *mut u8,
        mut off: usize,
        seq_len: usize,
        max_seq: usize,
    ) -> Result<usize> {
        let conv_bytes = 8192 * 4 * 2;
        let rec_bytes = 32 * 128 * 128 * 4;
        let kv_head_bytes = seq_len * 256 * 2;
        let kv_stride = max_seq * 256 * 2;
        for (i, l) in self.layers.iter().enumerate() {
            if l.is_linear {
                let (cs_ptr, rs_ptr) = if let Some(rb_slice) = rbs {
                    (rb_slice[i].conv_snaps, rb_slice[i].rec_final)
                } else {
                    (l.conv_state, l.rec_state)
                };
                cuda.dtoh_async(unsafe { dst.add(off) }, cs_ptr, conv_bytes)?;
                off += conv_bytes;
                cuda.dtoh_async(unsafe { dst.add(off) }, rs_ptr, rec_bytes)?;
                off += rec_bytes;
            } else {
                for h in 0..4 {
                    cuda.dtoh_async(unsafe { dst.add(off) }, l.k_cache + (h * kv_stride) as u64, kv_head_bytes)?;
                    off += kv_head_bytes;
                }
                for h in 0..4 {
                    cuda.dtoh_async(unsafe { dst.add(off) }, l.v_cache + (h * kv_stride) as u64, kv_head_bytes)?;
                    off += kv_head_bytes;
                }
            }
        }
        Ok(off)
    }

    fn enqueue_htod(&self, cuda: &CudaRuntime, src: *const u8, mut off: usize, seq_len: usize, max_seq: usize) -> Result<usize> {
        let conv_bytes = 8192 * 4 * 2;
        let rec_bytes = 32 * 128 * 128 * 4;
        let kv_head_bytes = seq_len * 256 * 2;
        let kv_stride = max_seq * 256 * 2;
        for l in &self.layers {
            if l.is_linear {
                cuda.htod_async(l.conv_state, unsafe { src.add(off) }, conv_bytes)?;
                off += conv_bytes;
                cuda.htod_async(l.rec_state, unsafe { src.add(off) }, rec_bytes)?;
                off += rec_bytes;
            } else {
                for h in 0..4 {
                    cuda.htod_async(l.k_cache + (h * kv_stride) as u64, unsafe { src.add(off) }, kv_head_bytes)?;
                    off += kv_head_bytes;
                }
                for h in 0..4 {
                    cuda.htod_async(l.v_cache + (h * kv_stride) as u64, unsafe { src.add(off) }, kv_head_bytes)?;
                    off += kv_head_bytes;
                }
            }
        }
        Ok(off)
    }
}

pub struct PrefillResult {
    pub first_tok: i64,
    pub cached_tokens: usize,
    pub cache_tier: &'static str,
    pub cache_restore_ms: f64,
    pub mtp_suffix_start: usize,
    pub deferred_hist_snap: usize, // if > 0, rb.rec_final/conv_snaps hold exact state at this token length!
}

pub struct StreamToken {
    pub token_id: i64,
    pub text: String,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub cache_tier: &'static str,
    pub cache_restore_ms: f64,
    pub completion_tokens: usize,
    pub finish_reason: Option<&'static str>,
}

#[derive(Clone, Copy)]
struct LinearRollbackPtrs {
    rec_final: CUdeviceptr,  // [32, 128, 128] f32
    conv_snaps: CUdeviceptr, // [M, 8192, 4] bf16
    saved_qkv: CUdeviceptr,  // [M, 8192] bf16
    saved_bg: CUdeviceptr,   // [M, 32, 2] f32
}

struct MtpRuntime {
    draft_k: usize,
    verify_m: usize,
    weights: MtpWeights,
    // MTP KV cache + Verify scratch + Rollback arena
    arena: CUdeviceptr,
    bytes: usize,
    mtp_k_cache: CUdeviceptr,
    mtp_v_cache: CUdeviceptr,
    // Verify M buffers (M = draft_k + 1 <= 4)
    v_tok_ids: CUdeviceptr,   // [4] i64
    v_out_toks: CUdeviceptr,  // [4] i64
    v_pos: CUdeviceptr,       // [1] i32
    v_hidden: CUdeviceptr,    // [4, 2560] bf16
    v_bridge: CUdeviceptr,    // [4, 2560] bf16
    v_normed: CUdeviceptr,    // [4, 2560] bf16 (also holds final H_main[0..M-1]!)
    v_delta: CUdeviceptr,     // [4, 2560] bf16
    v_dn_in: CUdeviceptr,     // [4, 12352] bf16
    v_dn_raw: CUdeviceptr,    // [4, 4096] f32
    v_core_out: CUdeviceptr,  // [4, 4096] bf16
    v_fa_qkv: CUdeviceptr,    // [4, 10240] bf16
    v_q_rot: CUdeviceptr,     // [4, 16, 256] bf16
    v_flash_out: CUdeviceptr, // [4, 16, 4, 256] f32
    v_flash_ml: CUdeviceptr,  // [4, 16, 4, 2] f32
    v_mlp_gu: CUdeviceptr,    // [4, 18432] bf16
    v_mlp_act: CUdeviceptr,   // [4, 9216] bf16
    v_logits: CUdeviceptr,    // [4, vocab_size] bf16
    v_arg_vals: CUdeviceptr,  // [4, 256] f32
    v_arg_idxs: CUdeviceptr,  // [4, 256] i32
    // MTP single/batch step buffers
    mtp_cat: CUdeviceptr,     // [4, 5120] bf16
    mtp_h: CUdeviceptr,       // [4, 2560] bf16
    mtp_h_norm: CUdeviceptr,  // [4, 2560] bf16
    mtp_tok_ids: CUdeviceptr, // [4] i64
    mtp_pos: CUdeviceptr,     // [1] i32
    mtp_out_tok: CUdeviceptr, // [1] i64
    // Per-stage DeltaNet rollback ptrs (indexed by layer_idx; dummy for full_attn)
    p1_rb: Vec<LinearRollbackPtrs>,
    p2_rb: Vec<LinearRollbackPtrs>,
    // Pinned host buffers
    h_verify_toks: *mut i64,
    h_draft_toks: *mut i64,
    h_mtp_tok: *mut i64,
    // Phase R7.1: Full-Cycle MTP CUDA Graphs (ONLY 2 graph launches per cycle, 0 eager kernels!)
    draft_and_verify_graph: CUgraphExec,
    commit_and_d1_graphs: Vec<CUgraphExec>, // indexed by accepted in 0..verify_m
    // Stats
    total_drafted: u64,
    total_accepted: u64,
    total_cycles: u64,
}

unsafe impl Send for MtpRuntime {}

pub struct RustSplitEngine {
    pub cuda: CudaRuntime,
    funcs: KernelFuncs,
    pub cfg: TextConfig,
    pub tok: QwenTokenizer,
    pub split_layer: usize,
    pub max_seq: usize,
    pub load_time_s: f64,

    weights: LoadedSplitWeights,
    p1_cache: StageCache,
    p2_cache: StageCache,

    // RoPE tables [max_seq, 64] bf16
    rope_arena: CUdeviceptr,
    cos_table: CUdeviceptr,
    sin_table: CUdeviceptr,

    // Single Decode Scratch Arena (M=2 Tensor-Core padded)
    scratch_arena: CUdeviceptr,
    scratch_bytes: usize,
    s_token_id: CUdeviceptr,
    s_next_token_id: CUdeviceptr,
    s_pos: CUdeviceptr,
    s_hidden: CUdeviceptr,     // [2, 2560]
    s_bridge: CUdeviceptr,     // [2, 2560] Zero-Copy GPU Inter-Stage Bridge!
    s_normed: CUdeviceptr,     // [2, 2560]
    s_delta: CUdeviceptr,      // [2, 2560]
    s_dn_in: CUdeviceptr,      // [2, 12352]
    s_qkv_conv: CUdeviceptr,   // [8192]
    s_core_out: CUdeviceptr,   // [2, 4096]
    s_fa_qkv: CUdeviceptr,     // [2, 10240]
    s_q_rot: CUdeviceptr,      // [16, 256]
    s_flash_out: CUdeviceptr,  // [16, 4, 256] f32
    s_flash_ml: CUdeviceptr,   // [16, 4, 2] f32
    s_mlp_gu: CUdeviceptr,     // [2, 18432]
    s_mlp_act: CUdeviceptr,    // [2, 9216]
    s_logits: CUdeviceptr,     // [2, vocab_size]
    s_arg_vals: CUdeviceptr,   // [256] f32
    s_arg_idxs: CUdeviceptr,   // [256] i32
    s_live_last_h: CUdeviceptr,// [2560] bf16 (5 KB on GPU for Tier 0 Hot-Cache MTP boundary!)

    // Reusable prefill scratch buffer (allocated once on demand up to max_seq)
    pf_arena: CUdeviceptr,
    pf_cap_seq: usize,

    // Pinned host ping-pong slots & CUDA events for Zero-Bubble SSE Overlap
    h_tok_slots: [*mut i64; 2],
    ev_slots: [CUevent; 2],

    decode_graph: CUgraphExec,

    // Optional Phase R3 MTP Speculative Decoding Runtime
    mtp: Option<MtpRuntime>,

    // Phase R6.1: toggle custom skinny GEMM on M=2..4 wide projections (dn_in, fa_qkv, mlp_gu) for A/B tuning
    skinny_m4_wide: bool,

    // Phase R4: 3-Tier Hybrid Prefix Cache (Tier 0 GPU Hot-Cache + Tier 1 Pinned-RAM Pool, 0 MiB extra VRAM!)
    live_tokens: Vec<i64>,
    live_has_mtp: bool,
    prefix_cache_max_slots: usize,
    prefix_cache: HashMap<Vec<i64>, PinnedSnapshot>,
    prefix_lru: VecDeque<Vec<i64>>,
    free_ram_pool: Vec<Vec<u8>>,
    pinned_staging: Option<PinnedBuf>,
    prefix_cache_hits: u64,
    prefix_cache_saved_tokens: u64,
}

unsafe impl Send for RustSplitEngine {}

fn f32_to_bf16_u16(v: f32) -> u16 {
    let mut u = v.to_bits();
    let lsb = (u >> 16) & 1;
    u = u.wrapping_add(0x7FFF + lsb);
    (u >> 16) as u16
}

pub(crate) fn build_rope_tables_bf16(cfg: &TextConfig, max_seq: usize) -> (Vec<u16>, Vec<u16>) {
    let rotary_dim = ((cfg.head_dim as f64) * cfg.rope_parameters.partial_rotary_factor) as usize; // 64
    let half = rotary_dim / 2; // 32
    let theta = cfg.rope_parameters.rope_theta;
    let mut inv_freq = vec![0.0f64; half];
    for i in 0..half {
        inv_freq[i] = 1.0 / theta.powf((2 * i) as f64 / rotary_dim as f64);
    }
    let mut cos_tbl = vec![0u16; max_seq * rotary_dim];
    let mut sin_tbl = vec![0u16; max_seq * rotary_dim];
    for p in 0..max_seq {
        let pf = p as f64;
        for i in 0..half {
            let ang = pf * inv_freq[i];
            let c = f32_to_bf16_u16(ang.cos() as f32);
            let s = f32_to_bf16_u16(ang.sin() as f32);
            cos_tbl[p * rotary_dim + i] = c;
            cos_tbl[p * rotary_dim + half + i] = c;
            sin_tbl[p * rotary_dim + i] = s;
            sin_tbl[p * rotary_dim + half + i] = s;
        }
    }
    (cos_tbl, sin_tbl)
}

impl RustSplitEngine {
    pub fn load(
        model_dir: &Path,
        split_layer: usize,
        mut max_seq: usize,
        mtp_drafts: usize,
        prefix_cache_slots: usize,
        force_max_seq: bool,
        skinny_m4_wide: bool,
    ) -> Result<Self> {
        let t0 = Instant::now();
        let cfg = TextConfig::load(model_dir)?;
        let tok = QwenTokenizer::load(model_dir, cfg.eos_token_id)?;
        let cuda = CudaRuntime::new(PTX_BYTES)?;
        let funcs = KernelFuncs::load(&cuda)?;

        let weights = load_split_weights(&cuda, model_dir, &cfg, split_layer, mtp_drafts > 0)?;

        // Phase R6.1 + R9: Bulletproof Windows WDDM VRAM Headroom Guard (prevents silent PCIe paging drops!)
        if !force_max_seq {
            let (free_b, _) = cuda.mem_info()?;
            let free_mib = free_b as f64 / 1048576.0;
            // Fixed allocations after weights: DeltaNet states (50) + Scratch (2) + MTP rollback (55) + pf_arena (47) + Graph/cuBLAS runtime (~70)
            let fixed_mib = 50.0 + 2.0 + if mtp_drafts > 0 { 55.0 } else { 0.0 } + 47.0 + 70.0;
            let kv_layers = if mtp_drafts > 0 { 9.0 } else { 8.0 };
            // Keep >= 450 MiB headroom at startup so opening Chrome/UI never crosses the ~380 MiB WDDM paging cliff!
            let wddm_guard_mib = 450.0;
            let kv_mib_for = |seq: usize| kv_layers * (seq as f64 / 1024.0) * 4.0;
            if free_mib - fixed_mib - kv_mib_for(max_seq) < wddm_guard_mib {
                let req = max_seq;
                let mut chosen = 1024usize;
                for &cand in &[2048usize, 1536, 1024] {
                    if cand < max_seq && free_mib - fixed_mib - kv_mib_for(cand) >= wddm_guard_mib {
                        chosen = cand;
                        break;
                    }
                }
                max_seq = chosen.min(req);
                println!(
                    "[RustSplitServer] WDDM VRAM Guard: free={:.1} MiB after weights -> clamped max_seq {} -> {} to keep >= {:.0} MiB headroom (pass --force-max-seq to override)",
                    free_mib, req, max_seq, wddm_guard_mib
                );
            }
        }

        let p1_cache = StageCache::alloc(&cuda, &weights.p1_layers, max_seq)?;
        let p2_cache = StageCache::alloc(&cuda, &weights.p2_layers, max_seq)?;

        // Upload RoPE tables [max_seq, 64] bf16
        let (cos_h, sin_h) = build_rope_tables_bf16(&cfg, max_seq);
        let tbl_bytes = cos_h.len() * 2;
        let tbl_aligned = (tbl_bytes + 255) & !255;
        let rope_arena = cuda.alloc(tbl_aligned * 2)?;
        let cos_table = rope_arena;
        let sin_table = rope_arena + tbl_aligned as u64;
        cuda.htod_sync(cos_table, cos_h.as_ptr() as *const u8, tbl_bytes)?;
        cuda.htod_sync(sin_table, sin_h.as_ptr() as *const u8, tbl_bytes)?;

        // Single Decode Scratch Arena (zero fragmentation!)
        let align = |b: usize| (b + 255) & !255;
        let mut off = 0usize;
        let mut next = |b: usize| {
            let cur = off;
            off += align(b);
            cur as u64
        };
        let o_tok = next(8);
        let o_ntok = next(8);
        let o_pos = next(4);
        let o_hid = next(2 * 2560 * 2);
        let o_br = next(2 * 2560 * 2);
        let o_nrm = next(2 * 2560 * 2);
        let o_del = next(2 * 2560 * 2);
        let o_dn = next(2 * 12352 * 2);
        let o_cv = next(8192 * 2);
        let o_co = next(2 * 4096 * 2);
        let o_fa = next(2 * 10240 * 2);
        let o_qr = next(16 * 256 * 2);
        let o_fo = next(16 * 4 * 256 * 4);
        let o_fm = next(16 * 4 * 2 * 4);
        let o_gu = next(2 * 18432 * 2);
        let o_ac = next(2 * 9216 * 2);
        let o_lg = next(2 * cfg.vocab_size * 2);
        let o_av = next(256 * 4);
        let o_ai = next(256 * 4);
        let o_lh = next(2560 * 2);

        let scratch_arena = cuda.alloc(off)?;
        cuda.memset_zero_async(scratch_arena, off)?;
        cuda.sync()?;

        let h0 = cuda.alloc_pinned(8)? as *mut i64;
        let h1 = cuda.alloc_pinned(8)? as *mut i64;
        let ev0 = cuda.create_event()?;
        let ev1 = cuda.create_event()?;

        // Optional Phase R3 MTP Runtime allocation
        let mtp_rt = if mtp_drafts > 0 {
            if let Some(mw) = weights.mtp {
                let dk = mtp_drafts.clamp(1, 3);
                let vm = dk + 1;
                let mut moff = 0usize;
                let mut mnext = |b: usize| {
                    let cur = moff;
                    moff += align(b);
                    cur as u64
                };
                let kv_sz = 4 * max_seq * 256 * 2;
                let o_mk = mnext(kv_sz);
                let o_mv = mnext(kv_sz);
                let o_vtok = mnext(4 * 8);
                let o_vout = mnext(4 * 8);
                let o_vpos = mnext(4);
                let o_vhid = mnext(4 * 2560 * 2);
                let o_vbr = mnext(4 * 2560 * 2);
                let o_vnrm = mnext(4 * 2560 * 2);
                let o_vdel = mnext(4 * 2560 * 2);
                let o_vdn = mnext(4 * 12352 * 2);
                let o_vraw = mnext(4 * 4096 * 4);
                let o_vco = mnext(4 * 4096 * 2);
                let o_vfa = mnext(4 * 10240 * 2);
                let o_vqr = mnext(4 * 16 * 256 * 2);
                let o_vfo = mnext(4 * 16 * 4 * 256 * 4);
                let o_vfm = mnext(4 * 16 * 4 * 2 * 4);
                let o_vgu = mnext(4 * 18432 * 2);
                let o_vac = mnext(4 * 9216 * 2);
                let o_vlg = mnext(4 * cfg.vocab_size * 2);
                let o_vav = mnext(4 * 256 * 4);
                let o_vai = mnext(4 * 256 * 4);
                let o_mcat = mnext(4 * 5120 * 2);
                let o_mh = mnext(4 * 2560 * 2);
                let o_mhn = mnext(4 * 2560 * 2);
                let o_mtid = mnext(4 * 8);
                let o_mpos = mnext(4);
                let o_mout = mnext(8);

                let mut p1_rb_off = Vec::with_capacity(weights.p1_layers.len());
                for l in &weights.p1_layers {
                    if l.is_linear {
                        p1_rb_off.push((
                            mnext(32 * 128 * 128 * 4),
                            mnext(vm * 8192 * 4 * 2),
                            mnext(vm * 8192 * 2),
                            mnext(vm * 32 * 2 * 4),
                        ));
                    } else {
                        p1_rb_off.push((0, 0, 0, 0));
                    }
                }
                let mut p2_rb_off = Vec::with_capacity(weights.p2_layers.len());
                for l in &weights.p2_layers {
                    if l.is_linear {
                        p2_rb_off.push((
                            mnext(32 * 128 * 128 * 4),
                            mnext(vm * 8192 * 4 * 2),
                            mnext(vm * 8192 * 2),
                            mnext(vm * 32 * 2 * 4),
                        ));
                    } else {
                        p2_rb_off.push((0, 0, 0, 0));
                    }
                }

                let arena = cuda.alloc(moff)?;
                cuda.memset_zero_async(arena, moff)?;
                cuda.sync()?;

                let p1_rb = weights.p1_layers.iter().zip(p1_rb_off.iter()).map(|(l, &(a, b, c, d))| {
                    if l.is_linear {
                        LinearRollbackPtrs {
                            rec_final: arena + a,
                            conv_snaps: arena + b,
                            saved_qkv: arena + c,
                            saved_bg: arena + d,
                        }
                    } else {
                        LinearRollbackPtrs { rec_final: 0, conv_snaps: 0, saved_qkv: 0, saved_bg: 0 }
                    }
                }).collect();
                let p2_rb = weights.p2_layers.iter().zip(p2_rb_off.iter()).map(|(l, &(a, b, c, d))| {
                    if l.is_linear {
                        LinearRollbackPtrs {
                            rec_final: arena + a,
                            conv_snaps: arena + b,
                            saved_qkv: arena + c,
                            saved_bg: arena + d,
                        }
                    } else {
                        LinearRollbackPtrs { rec_final: 0, conv_snaps: 0, saved_qkv: 0, saved_bg: 0 }
                    }
                }).collect();

                let h_verify_toks = cuda.alloc_pinned(4 * 8)? as *mut i64;
                let h_draft_toks = cuda.alloc_pinned(4 * 8)? as *mut i64;
                let h_mtp_tok = cuda.alloc_pinned(8)? as *mut i64;

                Some(MtpRuntime {
                    draft_k: dk,
                    verify_m: vm,
                    weights: mw,
                    arena,
                    bytes: moff,
                    mtp_k_cache: arena + o_mk,
                    mtp_v_cache: arena + o_mv,
                    v_tok_ids: arena + o_vtok,
                    v_out_toks: arena + o_vout,
                    v_pos: arena + o_vpos,
                    v_hidden: arena + o_vhid,
                    v_bridge: arena + o_vbr,
                    v_normed: arena + o_vnrm,
                    v_delta: arena + o_vdel,
                    v_dn_in: arena + o_vdn,
                    v_dn_raw: arena + o_vraw,
                    v_core_out: arena + o_vco,
                    v_fa_qkv: arena + o_vfa,
                    v_q_rot: arena + o_vqr,
                    v_flash_out: arena + o_vfo,
                    v_flash_ml: arena + o_vfm,
                    v_mlp_gu: arena + o_vgu,
                    v_mlp_act: arena + o_vac,
                    v_logits: arena + o_vlg,
                    v_arg_vals: arena + o_vav,
                    v_arg_idxs: arena + o_vai,
                    mtp_cat: arena + o_mcat,
                    mtp_h: arena + o_mh,
                    mtp_h_norm: arena + o_mhn,
                    mtp_tok_ids: arena + o_mtid,
                    mtp_pos: arena + o_mpos,
                    mtp_out_tok: arena + o_mout,
                    p1_rb,
                    p2_rb,
                    h_verify_toks,
                    h_draft_toks,
                    h_mtp_tok,
                    draft_and_verify_graph: std::ptr::null_mut(),
                    commit_and_d1_graphs: Vec::new(),
                    total_drafted: 0,
                    total_accepted: 0,
                    total_cycles: 0,
                })
            } else {
                None
            }
        } else {
            None
        };

        // Compact 16 MiB Pinned Host Staging Buffer (0 MiB WDDM VRAM penalty, batches ~54 MiB snapshot into ONLY 4 syncs!)
        let pinned_staging = if prefix_cache_slots > 0 {
            Some(PinnedBuf::alloc(&cuda, 16 * 1024 * 1024)?)
        } else {
            None
        };

        let mut eng = Self {
            cuda,
            funcs,
            cfg,
            tok,
            split_layer,
            max_seq,
            load_time_s: 0.0,
            weights,
            p1_cache,
            p2_cache,
            rope_arena,
            cos_table,
            sin_table,
            scratch_arena,
            scratch_bytes: off,
            s_token_id: scratch_arena + o_tok,
            s_next_token_id: scratch_arena + o_ntok,
            s_pos: scratch_arena + o_pos,
            s_hidden: scratch_arena + o_hid,
            s_bridge: scratch_arena + o_br,
            s_normed: scratch_arena + o_nrm,
            s_delta: scratch_arena + o_del,
            s_dn_in: scratch_arena + o_dn,
            s_qkv_conv: scratch_arena + o_cv,
            s_core_out: scratch_arena + o_co,
            s_fa_qkv: scratch_arena + o_fa,
            s_q_rot: scratch_arena + o_qr,
            s_flash_out: scratch_arena + o_fo,
            s_flash_ml: scratch_arena + o_fm,
            s_mlp_gu: scratch_arena + o_gu,
            s_mlp_act: scratch_arena + o_ac,
            s_logits: scratch_arena + o_lg,
            s_arg_vals: scratch_arena + o_av,
            s_arg_idxs: scratch_arena + o_ai,
            s_live_last_h: scratch_arena + o_lh,
            pf_arena: 0,
            pf_cap_seq: 0,
            h_tok_slots: [h0, h1],
            ev_slots: [ev0, ev1],
            decode_graph: std::ptr::null_mut(),
            mtp: mtp_rt,
            skinny_m4_wide,
            live_tokens: Vec::new(),
            live_has_mtp: false,
            prefix_cache_max_slots: prefix_cache_slots,
            prefix_cache: HashMap::new(),
            prefix_lru: VecDeque::new(),
            free_ram_pool: Vec::with_capacity(prefix_cache_slots),
            pinned_staging,
            prefix_cache_hits: 0,
            prefix_cache_saved_tokens: 0,
        };

        // Pre-allocate chunked prefill workspace (513 rows = 46.6 MiB) upfront so steady-state VRAM never spikes mid-chat!
        let _ = eng.ensure_prefill_workspace(513)?;

        eng.capture_decode_graph()?;
        if eng.mtp.is_some() {
            eng.capture_verify_graph()?;
        }
        eng.load_time_s = t0.elapsed().as_secs_f64();
        Ok(eng)
    }

    pub fn prefix_cache_stats(&self) -> (usize, usize, u64, u64, f64) {
        let ram_bytes: usize = self.prefix_cache.values().map(|e| e.data.len()).sum();
        (
            self.prefix_cache.len(),
            self.prefix_cache_max_slots,
            self.prefix_cache_hits,
            self.prefix_cache_saved_tokens,
            ram_bytes as f64 / (1024.0 * 1024.0),
        )
    }

    pub fn mtp_drafts(&self) -> usize {
        self.mtp.as_ref().map(|m| m.draft_k).unwrap_or(0)
    }

    pub fn mtp_stats(&self) -> (u64, u64, f64, f64) {
        if let Some(m) = &self.mtp {
            let acc_rate = if m.total_drafted > 0 {
                m.total_accepted as f64 / m.total_drafted as f64
            } else {
                0.0
            };
            let avg_toks = if m.total_cycles > 0 {
                (m.total_accepted + m.total_cycles) as f64 / m.total_cycles as f64
            } else {
                1.0
            };
            (m.total_accepted, m.total_drafted, acc_rate, avg_toks)
        } else {
            (0, 0, 0.0, 1.0)
        }
    }

    pub fn vram_allocated_mib(&self) -> f64 {
        let mtp_b = self.mtp.as_ref().map(|m| m.weights.bytes + m.bytes).unwrap_or(0);
        let total = self.weights.p1_bytes
            + self.weights.p2_bytes
            + self.p1_cache.bytes
            + self.p2_cache.bytes
            + self.scratch_bytes
            + (self.max_seq * 64 * 4)
            + mtp_b;
        total as f64 / (1024.0 * 1024.0)
    }

    pub fn prefix_cache_len(&self) -> usize {
        self.prefix_cache.len()
    }

    /// Phase R6.1: Dispatch 8-warp Coalesced Skinny BF16 GEMM (M=1..4) with optional Fused Residual Add.
    /// If `skinny_m4_wide` is false, routes wide projections (`N > 2560`, `M >= 2`, `!add_res`) to cuBLAS Tensor Cores!
    fn launch_skinny_gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        x: CUdeviceptr,
        w: CUdeviceptr,
        y: CUdeviceptr,
        add_res: bool,
    ) -> Result<()> {
        if m >= 2 && n > 2560 && !add_res && !self.skinny_m4_wide {
            return self.cuda.gemm_bf16_nt(m, n, k, x, w, y);
        }
        let mut x_p = x;
        let mut w_p = w;
        let mut y_p = y;
        let mut n_i: i32 = n as i32;
        let mut k_i: i32 = k as i32;
        let mut add_i: i32 = if add_res { 1 } else { 0 };
        let (func, grid_x) = match m {
            1 => (self.funcs.skinny_gemm_m1, ((n + 3) / 4) as u32),
            2 => (self.funcs.skinny_gemm_m2, ((n + 7) / 8) as u32),
            3 => (self.funcs.skinny_gemm_m3, ((n + 7) / 8) as u32),
            4 => (self.funcs.skinny_gemm_m4, ((n + 7) / 8) as u32),
            _ => unreachable!("launch_skinny_gemm supports M=1..4"),
        };
        self.cuda.launch(
            func,
            (grid_x, 1, 1),
            (256, 1, 1),
            &mut [
                &mut x_p as *mut _ as *mut c_void,
                &mut w_p as *mut _ as *mut c_void,
                &mut y_p as *mut _ as *mut c_void,
                &mut n_i as *mut _ as *mut c_void,
                &mut k_i as *mut _ as *mut c_void,
                &mut add_i as *mut _ as *mut c_void,
            ],
        )
    }

    fn run_stage_decode_ops(
        &self,
        layers: &[PackedLayerPtrs],
        caches: &[LayerCachePtrs],
        h_buf: CUdeviceptr,
    ) -> Result<()> {
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut c8192: i32 = 8192;
        let mut inter: i32 = 9216;
        let mut rows1: i32 = 1;
        let mut max_seq_i: i32 = self.max_seq as i32;
        let mut n_splits: i32 = 4;

        for (lyr, lc) in layers.iter().zip(caches.iter()) {
            // 1. Input RMSNorm
            let mut h_p = h_buf;
            let mut nw_p = lyr.in_norm;
            let mut out_p = self.s_normed;
            self.cuda.launch(
                self.funcs.rmsnorm_2d,
                (1, 1, 1),
                (256, 1, 1),
                &mut [
                    &mut h_p as *mut _ as *mut c_void,
                    &mut nw_p as *mut _ as *mut c_void,
                    &mut out_p as *mut _ as *mut c_void,
                    &mut dim as *mut _ as *mut c_void,
                    &mut eps as *mut _ as *mut c_void,
                ],
            )?;

            if lyr.is_linear {
                self.launch_skinny_gemm(1, 12352, 2560, self.s_normed, lyr.in_all, self.s_dn_in, false)?;
                let mut x_in = self.s_dn_in;
                let mut cs = lc.conv_state;
                let mut cw = lyr.conv_w;
                let mut cv_out = self.s_qkv_conv;
                self.cuda.launch(
                    self.funcs.conv1d_update,
                    (32, 1, 1),
                    (256, 1, 1),
                    &mut [
                        &mut x_in as *mut _ as *mut c_void,
                        &mut cs as *mut _ as *mut c_void,
                        &mut cw as *mut _ as *mut c_void,
                        &mut cv_out as *mut _ as *mut c_void,
                        &mut c8192 as *mut _ as *mut c_void,
                    ],
                )?;

                let mut pin = self.s_dn_in;
                let mut dtb = lyr.dt_bias;
                let mut alog = lyr.a_log_f32;
                let mut dnw = lyr.dn_norm;
                let mut rec = lc.rec_state;
                let mut co = self.s_core_out;
                self.cuda.launch(
                    self.funcs.deltanet_decode_packed,
                    (32, 1, 1),
                    (256, 1, 1),
                    &mut [
                        &mut cv_out as *mut _ as *mut c_void,
                        &mut pin as *mut _ as *mut c_void,
                        &mut dtb as *mut _ as *mut c_void,
                        &mut alog as *mut _ as *mut c_void,
                        &mut dnw as *mut _ as *mut c_void,
                        &mut rec as *mut _ as *mut c_void,
                        &mut co as *mut _ as *mut c_void,
                        &mut eps as *mut _ as *mut c_void,
                    ],
                )?;
                self.launch_skinny_gemm(1, 2560, 4096, self.s_core_out, lyr.out_proj, self.s_delta, false)?;
            } else {
                self.launch_skinny_gemm(1, 10240, 2560, self.s_normed, lyr.qkv, self.s_fa_qkv, false)?;
                let mut qkv_p = self.s_fa_qkv;
                let mut qnw = lyr.q_norm;
                let mut knw = lyr.k_norm;
                let mut cos_p = self.cos_table;
                let mut sin_p = self.sin_table;
                let mut pos_p = self.s_pos;
                let mut qrot = self.s_q_rot;
                let mut kc = lc.k_cache;
                let mut vc = lc.v_cache;
                self.cuda.launch(
                    self.funcs.qk_rope_kv_write_packed,
                    (20, 1, 1),
                    (256, 1, 1),
                    &mut [
                        &mut qkv_p as *mut _ as *mut c_void,
                        &mut qnw as *mut _ as *mut c_void,
                        &mut knw as *mut _ as *mut c_void,
                        &mut cos_p as *mut _ as *mut c_void,
                        &mut sin_p as *mut _ as *mut c_void,
                        &mut pos_p as *mut _ as *mut c_void,
                        &mut qrot as *mut _ as *mut c_void,
                        &mut kc as *mut _ as *mut c_void,
                        &mut vc as *mut _ as *mut c_void,
                        &mut max_seq_i as *mut _ as *mut c_void,
                        &mut eps as *mut _ as *mut c_void,
                    ],
                )?;

                let mut fo = self.s_flash_out;
                let mut fm = self.s_flash_ml;
                self.cuda.launch(
                    self.funcs.gqa_flash_splitk,
                    (16, 4, 1),
                    (256, 1, 1),
                    &mut [
                        &mut qrot as *mut _ as *mut c_void,
                        &mut kc as *mut _ as *mut c_void,
                        &mut vc as *mut _ as *mut c_void,
                        &mut pos_p as *mut _ as *mut c_void,
                        &mut fo as *mut _ as *mut c_void,
                        &mut fm as *mut _ as *mut c_void,
                        &mut max_seq_i as *mut _ as *mut c_void,
                        &mut n_splits as *mut _ as *mut c_void,
                    ],
                )?;
                let mut co = self.s_core_out;
                self.cuda.launch(
                    self.funcs.gqa_flash_reduce,
                    (16, 1, 1),
                    (256, 1, 1),
                    &mut [
                        &mut fo as *mut _ as *mut c_void,
                        &mut fm as *mut _ as *mut c_void,
                        &mut qkv_p as *mut _ as *mut c_void,
                        &mut co as *mut _ as *mut c_void,
                        &mut n_splits as *mut _ as *mut c_void,
                    ],
                )?;
                self.launch_skinny_gemm(1, 2560, 4096, self.s_core_out, lyr.o_proj, self.s_delta, false)?;
            }

            // 2. Residual Add + Post-Attention RMSNorm
            let mut del_p = self.s_delta;
            let mut pnw_p = lyr.post_norm;
            self.cuda.launch(
                self.funcs.add_rmsnorm_2d,
                (1, 1, 1),
                (256, 1, 1),
                &mut [
                    &mut h_p as *mut _ as *mut c_void,
                    &mut del_p as *mut _ as *mut c_void,
                    &mut pnw_p as *mut _ as *mut c_void,
                    &mut out_p as *mut _ as *mut c_void,
                    &mut dim as *mut _ as *mut c_void,
                    &mut eps as *mut _ as *mut c_void,
                ],
            )?;

            // 3. Packed SwiGLU MLP + Fused Residual Add into h_buf (zero residual_add kernel!)
            self.launch_skinny_gemm(1, 18432, 2560, self.s_normed, lyr.gate_up, self.s_mlp_gu, false)?;
            let mut gu_p = self.s_mlp_gu;
            let mut act_p = self.s_mlp_act;
            self.cuda.launch(
                self.funcs.silu_mul_packed_2d,
                (36, 1, 1),
                (256, 1, 1),
                &mut [
                    &mut gu_p as *mut _ as *mut c_void,
                    &mut act_p as *mut _ as *mut c_void,
                    &mut inter as *mut _ as *mut c_void,
                    &mut rows1 as *mut _ as *mut c_void,
                ],
            )?;
            self.launch_skinny_gemm(1, 2560, 9216, self.s_mlp_act, lyr.down, h_buf, true)?;
        }
        Ok(())
    }

    fn decode_body(&self) -> Result<()> {
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut vocab_i: i32 = self.cfg.vocab_size as i32;
        let mut nblk: i32 = 256;

        // ---- PART 1 (Stage 0) ----
        let mut tid_p = self.s_token_id;
        let mut emb_p = self.weights.embed_w;
        let mut hid_p = self.s_hidden;
        self.cuda.launch(
            self.funcs.embed_single,
            (10, 1, 1),
            (256, 1, 1),
            &mut [
                &mut tid_p as *mut _ as *mut c_void,
                &mut emb_p as *mut _ as *mut c_void,
                &mut hid_p as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
            ],
        )?;
        self.run_stage_decode_ops(&self.weights.p1_layers, &self.p1_cache.layers, self.s_hidden)?;

        // ---- ZERO-COPY GPU INTER-STAGE BRIDGE (Part 1 -> Part 2) ----
        let mut br_p = self.s_bridge;
        self.cuda.launch(
            self.funcs.copy_bf16,
            (10, 1, 1),
            (256, 1, 1),
            &mut [
                &mut hid_p as *mut _ as *mut c_void,
                &mut br_p as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
            ],
        )?;

        // ---- PART 2 (Stage 1) ----
        self.run_stage_decode_ops(&self.weights.p2_layers, &self.p2_cache.layers, self.s_bridge)?;

        let mut fnw = self.weights.final_norm_w;
        let mut nrm = self.s_normed;
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
        self.cuda.gemm_bf16_nt(2, self.cfg.vocab_size, 2560, self.s_normed, self.weights.lm_head_w, self.s_logits)?;

        // 2-Stage GPU Argmax + Auto-feedback s_next_token_id -> s_token_id + inc_pos
        let mut lg_p = self.s_logits;
        let mut av_p = self.s_arg_vals;
        let mut ai_p = self.s_arg_idxs;
        self.cuda.launch(
            self.funcs.argmax_stage1,
            (256, 1, 1),
            (256, 1, 1),
            &mut [
                &mut lg_p as *mut _ as *mut c_void,
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut vocab_i as *mut _ as *mut c_void,
            ],
        )?;
        let mut ntok_p = self.s_next_token_id;
        self.cuda.launch(
            self.funcs.argmax_stage2,
            (1, 1, 1),
            (256, 1, 1),
            &mut [
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut ntok_p as *mut _ as *mut c_void,
                &mut nblk as *mut _ as *mut c_void,
            ],
        )?;
        self.cuda.launch(
            self.funcs.copy_i64,
            (1, 1, 1),
            (1, 1, 1),
            &mut [
                &mut ntok_p as *mut _ as *mut c_void,
                &mut tid_p as *mut _ as *mut c_void,
            ],
        )?;
        let mut pos_p = self.s_pos;
        self.cuda.launch(
            self.funcs.inc_pos,
            (1, 1, 1),
            (1, 1, 1),
            &mut [&mut pos_p as *mut _ as *mut c_void],
        )?;
        Ok(())
    }

    fn capture_decode_graph(&mut self) -> Result<()> {
        let one_i64: i64 = 1;
        let pos_init: i32 = 8;
        self.cuda.htod_sync(self.s_token_id, &one_i64 as *const i64 as *const u8, 8)?;
        self.cuda.htod_sync(self.s_pos, &pos_init as *const i32 as *const u8, 4)?;
        for _ in 0..3 {
            self.decode_body()?;
        }
        self.cuda.sync()?;

        self.cuda.begin_capture()?;
        self.decode_body()?;
        self.decode_graph = self.cuda.end_capture_and_instantiate()?;
        self.cuda.sync()?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Phase R3: Parallel Verify (M = K + 1 <= 4) CUDA Graph + MTP Draft Head
    // -----------------------------------------------------------------------
    fn run_stage_verify_ops(
        &self,
        mtp: &MtpRuntime,
        layers: &[PackedLayerPtrs],
        caches: &[LayerCachePtrs],
        rbs: &[LinearRollbackPtrs],
        h_buf: CUdeviceptr,
    ) -> Result<()> {
        let m = mtp.verify_m;
        let mut m_i: i32 = m as i32;
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut c8192: i32 = 8192;
        let mut inter: i32 = 9216;
        let mut max_seq_i: i32 = self.max_seq as i32;
        let mut n_splits: i32 = 4;

        for ((lyr, lc), rb) in layers.iter().zip(caches.iter()).zip(rbs.iter()) {
            let mut h_p = h_buf;
            let mut nw_p = lyr.in_norm;
            let mut nrm_p = mtp.v_normed;
            self.cuda.launch(
                self.funcs.rmsnorm_2d,
                (m as u32, 1, 1),
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
                self.launch_skinny_gemm(m, 12352, 2560, mtp.v_normed, lyr.in_all, mtp.v_dn_in, false)?;
                let mut pin = mtp.v_dn_in;
                let mut cs = lc.conv_state;
                let mut csnaps = rb.conv_snaps;
                let mut cw = lyr.conv_w;
                let mut cv_out = rb.saved_qkv;
                self.cuda.launch(
                    self.funcs.conv1d_verify_m,
                    (32, 1, 1),
                    (256, 1, 1),
                    &mut [
                        &mut pin as *mut _ as *mut c_void,
                        &mut cs as *mut _ as *mut c_void,
                        &mut csnaps as *mut _ as *mut c_void,
                        &mut cw as *mut _ as *mut c_void,
                        &mut cv_out as *mut _ as *mut c_void,
                        &mut m_i as *mut _ as *mut c_void,
                        &mut c8192 as *mut _ as *mut c_void,
                    ],
                )?;

                let mut dtb = lyr.dt_bias;
                let mut alog = lyr.a_log_f32;
                let mut rec_in = lc.rec_state;
                let mut rec_fin = rb.rec_final;
                let mut raw_p = mtp.v_dn_raw;
                let mut sbg = rb.saved_bg;
                self.cuda.launch(
                    self.funcs.deltanet_verify_m,
                    (32, 4, 1),
                    (128, 1, 1),
                    &mut [
                        &mut cv_out as *mut _ as *mut c_void,
                        &mut pin as *mut _ as *mut c_void,
                        &mut dtb as *mut _ as *mut c_void,
                        &mut alog as *mut _ as *mut c_void,
                        &mut rec_in as *mut _ as *mut c_void,
                        &mut rec_fin as *mut _ as *mut c_void,
                        &mut raw_p as *mut _ as *mut c_void,
                        &mut sbg as *mut _ as *mut c_void,
                        &mut m_i as *mut _ as *mut c_void,
                    ],
                )?;

                let mut dnw = lyr.dn_norm;
                let mut co = mtp.v_core_out;
                self.cuda.launch(
                    self.funcs.deltanet_prefill_norm_gate,
                    (32, m as u32, 1),
                    (128, 1, 1),
                    &mut [
                        &mut raw_p as *mut _ as *mut c_void,
                        &mut pin as *mut _ as *mut c_void,
                        &mut dnw as *mut _ as *mut c_void,
                        &mut co as *mut _ as *mut c_void,
                        &mut eps as *mut _ as *mut c_void,
                    ],
                )?;
                self.launch_skinny_gemm(m, 2560, 4096, mtp.v_core_out, lyr.out_proj, mtp.v_delta, false)?;
            } else {
                self.launch_skinny_gemm(m, 10240, 2560, mtp.v_normed, lyr.qkv, mtp.v_fa_qkv, false)?;
                let mut qkv_p = mtp.v_fa_qkv;
                let mut qnw = lyr.q_norm;
                let mut knw = lyr.k_norm;
                let mut cos_p = self.cos_table;
                let mut sin_p = self.sin_table;
                let mut pos_p = mtp.v_pos;
                let mut qrot = mtp.v_q_rot;
                let mut kc = lc.k_cache;
                let mut vc = lc.v_cache;
                self.cuda.launch(
                    self.funcs.qk_rope_kv_verify_m,
                    (20, m as u32, 1),
                    (256, 1, 1),
                    &mut [
                        &mut qkv_p as *mut _ as *mut c_void,
                        &mut qnw as *mut _ as *mut c_void,
                        &mut knw as *mut _ as *mut c_void,
                        &mut cos_p as *mut _ as *mut c_void,
                        &mut sin_p as *mut _ as *mut c_void,
                        &mut pos_p as *mut _ as *mut c_void,
                        &mut qrot as *mut _ as *mut c_void,
                        &mut kc as *mut _ as *mut c_void,
                        &mut vc as *mut _ as *mut c_void,
                        &mut max_seq_i as *mut _ as *mut c_void,
                        &mut eps as *mut _ as *mut c_void,
                    ],
                )?;

                let mut fo = mtp.v_flash_out;
                let mut fm = mtp.v_flash_ml;
                self.cuda.launch(
                    self.funcs.gqa_verify_m_splitk,
                    (16, 4, m as u32),
                    (256, 1, 1),
                    &mut [
                        &mut qrot as *mut _ as *mut c_void,
                        &mut kc as *mut _ as *mut c_void,
                        &mut vc as *mut _ as *mut c_void,
                        &mut pos_p as *mut _ as *mut c_void,
                        &mut fo as *mut _ as *mut c_void,
                        &mut fm as *mut _ as *mut c_void,
                        &mut max_seq_i as *mut _ as *mut c_void,
                        &mut n_splits as *mut _ as *mut c_void,
                    ],
                )?;
                let mut co = mtp.v_core_out;
                self.cuda.launch(
                    self.funcs.gqa_verify_m_reduce,
                    (16, m as u32, 1),
                    (256, 1, 1),
                    &mut [
                        &mut fo as *mut _ as *mut c_void,
                        &mut fm as *mut _ as *mut c_void,
                        &mut qkv_p as *mut _ as *mut c_void,
                        &mut co as *mut _ as *mut c_void,
                        &mut n_splits as *mut _ as *mut c_void,
                    ],
                )?;
                self.launch_skinny_gemm(m, 2560, 4096, mtp.v_core_out, lyr.o_proj, mtp.v_delta, false)?;
            }

            let mut del_p = mtp.v_delta;
            let mut pnw = lyr.post_norm;
            self.cuda.launch(
                self.funcs.add_rmsnorm_2d,
                (m as u32, 1, 1),
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

            self.launch_skinny_gemm(m, 18432, 2560, mtp.v_normed, lyr.gate_up, mtp.v_mlp_gu, false)?;
            let mut gu_p = mtp.v_mlp_gu;
            let mut act_p = mtp.v_mlp_act;
            self.cuda.launch(
                self.funcs.silu_mul_packed_2d,
                (36, m as u32, 1),
                (256, 1, 1),
                &mut [
                    &mut gu_p as *mut _ as *mut c_void,
                    &mut act_p as *mut _ as *mut c_void,
                    &mut inter as *mut _ as *mut c_void,
                    &mut m_i as *mut _ as *mut c_void,
                ],
            )?;
            self.launch_skinny_gemm(m, 2560, 9216, mtp.v_mlp_act, lyr.down, h_buf, true)?;
        }
        Ok(())
    }

    fn verify_body(&self, mtp: &MtpRuntime) -> Result<()> {
        let m = mtp.verify_m;
        let mut m_i: i32 = m as i32;
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut vocab_i: i32 = self.cfg.vocab_size as i32;
        let mut nblk: i32 = 256;
        let mut total_h: i32 = (m * 2560) as i32;

        // PART 1 (Stage 0)
        let mut tid_p = mtp.v_tok_ids;
        let mut emb_p = self.weights.embed_w;
        let mut hid_p = mtp.v_hidden;
        self.cuda.launch(
            self.funcs.embed_prefill,
            (10, m as u32, 1),
            (256, 1, 1),
            &mut [
                &mut tid_p as *mut _ as *mut c_void,
                &mut emb_p as *mut _ as *mut c_void,
                &mut hid_p as *mut _ as *mut c_void,
                &mut m_i as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
            ],
        )?;
        self.run_stage_verify_ops(mtp, &self.weights.p1_layers, &self.p1_cache.layers, &mtp.p1_rb, mtp.v_hidden)?;

        // ZERO-COPY GPU INTER-STAGE BRIDGE (Part 1 -> Part 2)
        let mut br_p = mtp.v_bridge;
        let blocks = ((m * 2560 + 255) / 256) as u32;
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
        self.run_stage_verify_ops(mtp, &self.weights.p2_layers, &self.p2_cache.layers, &mtp.p2_rb, mtp.v_bridge)?;

        // Final RMSNorm -> mtp.v_normed holds post-norm H_main[0..M-1] for MTP!
        let mut fnw = self.weights.final_norm_w;
        let mut nrm = mtp.v_normed;
        self.cuda.launch(
            self.funcs.rmsnorm_2d,
            (m as u32, 1, 1),
            (256, 1, 1),
            &mut [
                &mut br_p as *mut _ as *mut c_void,
                &mut fnw as *mut _ as *mut c_void,
                &mut nrm as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;
        self.cuda.gemm_bf16_nt(m, self.cfg.vocab_size, 2560, mtp.v_normed, self.weights.lm_head_w, mtp.v_logits)?;

        // Batch Argmax across all M verified rows
        let mut lg_p = mtp.v_logits;
        let mut av_p = mtp.v_arg_vals;
        let mut ai_p = mtp.v_arg_idxs;
        let mut out_p = mtp.v_out_toks;
        self.cuda.launch(
            self.funcs.argmax_batch_stage1,
            (256, m as u32, 1),
            (256, 1, 1),
            &mut [
                &mut lg_p as *mut _ as *mut c_void,
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut vocab_i as *mut _ as *mut c_void,
            ],
        )?;
        self.cuda.launch(
            self.funcs.argmax_batch_stage2,
            (1, m as u32, 1),
            (256, 1, 1),
            &mut [
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut out_p as *mut _ as *mut c_void,
                &mut nblk as *mut _ as *mut c_void,
            ],
        )?;
        Ok(())
    }

    fn draft_and_verify_body(&self, mtp: &MtpRuntime) -> Result<()> {
        // Assumes v_tok_ids[0] = cur_tok, mtp.mtp_pos = cur_pos, mtp.v_pos = cur_pos, mtp.mtp_out_tok = d1!
        self.cuda.dtod_async(mtp.v_tok_ids + 8, mtp.mtp_out_tok, 8)?;
        let mut mpos_p = mtp.mtp_pos;
        for idx in 1..mtp.draft_k {
            self.cuda.dtod_async(mtp.v_hidden, mtp.mtp_h_norm, 2560 * 2)?;
            self.run_mtp_step_small_body(1, mtp.mtp_out_tok, mtp.v_hidden)?;
            self.cuda.dtod_async(mtp.v_tok_ids + ((idx + 1) * 8) as u64, mtp.mtp_out_tok, 8)?;
            if idx + 1 < mtp.draft_k {
                self.cuda.launch(
                    self.funcs.inc_pos,
                    (1, 1, 1),
                    (1, 1, 1),
                    &mut [&mut mpos_p as *mut _ as *mut c_void],
                )?;
            }
        }
        self.cuda.dtoh_async(mtp.h_draft_toks as *mut u8, mtp.v_tok_ids + 8, mtp.draft_k * 8)?;
        self.verify_body(mtp)?;
        self.cuda.dtoh_async(mtp.h_verify_toks as *mut u8, mtp.v_out_toks, mtp.verify_m * 8)?;
        Ok(())
    }

    fn capture_verify_graph(&mut self) -> Result<()> {
        let (verify_m, v_tok_ids, v_pos, mtp_tok_ids, mtp_pos, mtp_out_tok) = {
            let mtp = self.mtp.as_ref().unwrap();
            (mtp.verify_m, mtp.v_tok_ids, mtp.v_pos, mtp.mtp_tok_ids, mtp.mtp_pos, mtp.mtp_out_tok)
        };
        let toks = [1i64; 4];
        let one_tok: i64 = 1;
        let pos_init: i32 = 8;
        self.cuda.htod_sync(v_tok_ids, toks.as_ptr() as *const u8, 4 * 8)?;
        self.cuda.htod_sync(mtp_tok_ids, toks.as_ptr() as *const u8, 4 * 8)?;
        self.cuda.htod_sync(mtp_out_tok, &one_tok as *const i64 as *const u8, 8)?;
        self.cuda.htod_sync(v_pos, &pos_init as *const i32 as *const u8, 4)?;
        self.cuda.htod_sync(mtp_pos, &pos_init as *const i32 as *const u8, 4)?;

        // 1. Capture Unified `draft_and_verify_graph` (d2..dK + Verify M=4 + async D2H of drafts & verified toks!)
        for _ in 0..2 {
            self.cuda.htod_sync(mtp_pos, &pos_init as *const i32 as *const u8, 4)?;
            let mtp = self.mtp.as_ref().unwrap();
            self.draft_and_verify_body(mtp)?;
        }
        self.cuda.sync()?;
        self.cuda.htod_sync(mtp_pos, &pos_init as *const i32 as *const u8, 4)?;
        self.cuda.begin_capture()?;
        {
            let mtp = self.mtp.as_ref().unwrap();
            self.draft_and_verify_body(mtp)?;
        }
        let dv_exec = self.cuda.end_capture_and_instantiate()?;
        self.cuda.sync()?;

        // 2. Capture `commit_and_d1_graphs[accepted]` for accepted = 0..verify_m (Commit DeltaNet + MTP d1!)
        let mut c_graphs = Vec::with_capacity(verify_m);
        for accepted in 0..verify_m {
            let num_new = accepted + 1;
            let (mtid, vnrm) = {
                let mtp = self.mtp.as_ref().unwrap();
                (mtp.mtp_tok_ids, mtp.v_normed)
            };
            for _ in 0..2 {
                self.cuda.htod_sync(mtp_pos, &pos_init as *const i32 as *const u8, 4)?;
                self.commit_deltanet_states(accepted)?;
                self.run_mtp_step_small_body(num_new, mtid, vnrm)?;
            }
            self.cuda.sync()?;
            self.cuda.htod_sync(mtp_pos, &pos_init as *const i32 as *const u8, 4)?;
            self.cuda.begin_capture()?;
            self.commit_deltanet_states(accepted)?;
            self.run_mtp_step_small_body(num_new, mtid, vnrm)?;
            let exec = self.cuda.end_capture_and_instantiate()?;
            self.cuda.sync()?;
            c_graphs.push(exec);
        }

        let mtp_mut = self.mtp.as_mut().unwrap();
        mtp_mut.draft_and_verify_graph = dv_exec;
        mtp_mut.commit_and_d1_graphs = c_graphs;
        Ok(())
    }

    fn commit_deltanet_states(&self, accepted: usize) -> Result<()> {
        let mtp = self.mtp.as_ref().unwrap();
        let num_steps = accepted + 1; // 1..=M
        let full_accept = num_steps == mtp.verify_m;
        let mut steps_i: i32 = num_steps as i32;
        let conv_bytes = 8192 * 4 * 2;
        let rec_bytes = 32 * 128 * 128 * 4;

        let mut commit_stage = |caches: &[LayerCachePtrs], rbs: &[LinearRollbackPtrs]| -> Result<()> {
            for (lc, rb) in caches.iter().zip(rbs.iter()) {
                if !lc.is_linear {
                    continue;
                }
                let snap_ptr = rb.conv_snaps + (accepted * conv_bytes) as u64;
                self.cuda.dtod_async(lc.conv_state, snap_ptr, conv_bytes)?;
                if full_accept {
                    self.cuda.dtod_async(lc.rec_state, rb.rec_final, rec_bytes)?;
                } else {
                    let mut qkv = rb.saved_qkv;
                    let mut sbg = rb.saved_bg;
                    let mut rec = lc.rec_state;
                    self.cuda.launch(
                        self.funcs.deltanet_commit_partial,
                        (32, 4, 1),
                        (128, 1, 1),
                        &mut [
                            &mut qkv as *mut _ as *mut c_void,
                            &mut sbg as *mut _ as *mut c_void,
                            &mut rec as *mut _ as *mut c_void,
                            &mut steps_i as *mut _ as *mut c_void,
                        ],
                    )?;
                }
            }
            Ok(())
        };
        commit_stage(&self.p1_cache.layers, &mtp.p1_rb)?;
        commit_stage(&self.p2_cache.layers, &mtp.p2_rb)?;
        Ok(())
    }

    /// Graph-captureable 1-layer MTP forward body for `m` tokens (`1..=4`) reading `mtp.mtp_pos` on GPU.
    fn run_mtp_step_small_body(
        &self,
        m: usize,
        tok_ids_gpu: CUdeviceptr,
        hid_gpu: CUdeviceptr,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().unwrap();
        let mw = mtp.weights;
        let lyr = mw.layer;
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut max_seq_i: i32 = self.max_seq as i32;
        let mut n_splits: i32 = 4;
        let mut inter: i32 = 9216;
        let mut m_i: i32 = m as i32;
        let mut vocab_i: i32 = self.cfg.vocab_size as i32;
        let mut nblk: i32 = 256;

        // 1. Fused embed(tok_ids) + pre_fc_norm_emb + pre_fc_norm_hid + concat -> mtp_cat [m, 5120]
        let mut tid_p = tok_ids_gpu;
        let mut emb_p = self.weights.embed_w;
        let mut hin_p = hid_gpu;
        let mut w_en = mw.pre_fc_norm_emb;
        let mut w_hn = mw.pre_fc_norm_hid;
        let mut cat_p = mtp.mtp_cat;
        self.cuda.launch(
            self.funcs.mtp_pre_fc_concat,
            (m as u32, 1, 1),
            (256, 1, 1),
            &mut [
                &mut tid_p as *mut _ as *mut c_void,
                &mut emb_p as *mut _ as *mut c_void,
                &mut hin_p as *mut _ as *mut c_void,
                &mut w_en as *mut _ as *mut c_void,
                &mut w_hn as *mut _ as *mut c_void,
                &mut cat_p as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;

        // 2. mtp.fc: [m, 5120] -> mtp_h [m, 2560] (Phase R6: Custom Skinny BF16 GEMM, no M=1->2 padding!)
        self.launch_skinny_gemm(m, 2560, 5120, mtp.mtp_cat, mw.fc_w, mtp.mtp_h, false)?;

        // 3. mtp.layers.0 (Full Attention + MLP)
        let mut h_p = mtp.mtp_h;
        let mut nw_p = lyr.in_norm;
        let mut nrm_p = mtp.v_delta; // temp normed buffer (doesn't touch v_normed!)
        self.cuda.launch(
            self.funcs.rmsnorm_2d,
            (m as u32, 1, 1),
            (256, 1, 1),
            &mut [
                &mut h_p as *mut _ as *mut c_void,
                &mut nw_p as *mut _ as *mut c_void,
                &mut nrm_p as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;

        self.launch_skinny_gemm(m, 10240, 2560, mtp.v_delta, lyr.qkv, mtp.v_fa_qkv, false)?;
        let mut qkv_p = mtp.v_fa_qkv;
        let mut qnw = lyr.q_norm;
        let mut knw = lyr.k_norm;
        let mut cos_p = self.cos_table;
        let mut sin_p = self.sin_table;
        let mut pos_p = mtp.mtp_pos;
        let mut qrot = mtp.v_q_rot;
        let mut kc = mtp.mtp_k_cache;
        let mut vc = mtp.mtp_v_cache;
        self.cuda.launch(
            self.funcs.qk_rope_kv_verify_m,
            (20, m as u32, 1),
            (256, 1, 1),
            &mut [
                &mut qkv_p as *mut _ as *mut c_void,
                &mut qnw as *mut _ as *mut c_void,
                &mut knw as *mut _ as *mut c_void,
                &mut cos_p as *mut _ as *mut c_void,
                &mut sin_p as *mut _ as *mut c_void,
                &mut pos_p as *mut _ as *mut c_void,
                &mut qrot as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut vc as *mut _ as *mut c_void,
                &mut max_seq_i as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;

        let mut fo = mtp.v_flash_out;
        let mut fm = mtp.v_flash_ml;
        self.cuda.launch(
            self.funcs.gqa_verify_m_splitk,
            (16, 4, m as u32),
            (256, 1, 1),
            &mut [
                &mut qrot as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut vc as *mut _ as *mut c_void,
                &mut pos_p as *mut _ as *mut c_void,
                &mut fo as *mut _ as *mut c_void,
                &mut fm as *mut _ as *mut c_void,
                &mut max_seq_i as *mut _ as *mut c_void,
                &mut n_splits as *mut _ as *mut c_void,
            ],
        )?;
        let mut co = mtp.v_core_out;
        self.cuda.launch(
            self.funcs.gqa_verify_m_reduce,
            (16, m as u32, 1),
            (256, 1, 1),
            &mut [
                &mut fo as *mut _ as *mut c_void,
                &mut fm as *mut _ as *mut c_void,
                &mut qkv_p as *mut _ as *mut c_void,
                &mut co as *mut _ as *mut c_void,
                &mut n_splits as *mut _ as *mut c_void,
            ],
        )?;
        self.launch_skinny_gemm(m, 2560, 4096, mtp.v_core_out, lyr.o_proj, mtp.v_delta, false)?;

        let mut del_p = mtp.v_delta;
        let mut pnw = lyr.post_norm;
        let mut nrm2 = mtp.v_bridge; // temp buffer (safe!)
        self.cuda.launch(
            self.funcs.add_rmsnorm_2d,
            (m as u32, 1, 1),
            (256, 1, 1),
            &mut [
                &mut h_p as *mut _ as *mut c_void,
                &mut del_p as *mut _ as *mut c_void,
                &mut pnw as *mut _ as *mut c_void,
                &mut nrm2 as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;

        self.launch_skinny_gemm(m, 18432, 2560, mtp.v_bridge, lyr.gate_up, mtp.v_mlp_gu, false)?;
        let mut gu_p = mtp.v_mlp_gu;
        let mut act_p = mtp.v_mlp_act;
        self.cuda.launch(
            self.funcs.silu_mul_packed_2d,
            (36, m as u32, 1),
            (256, 1, 1),
            &mut [
                &mut gu_p as *mut _ as *mut c_void,
                &mut act_p as *mut _ as *mut c_void,
                &mut inter as *mut _ as *mut c_void,
                &mut m_i as *mut _ as *mut c_void,
            ],
        )?;
        self.launch_skinny_gemm(m, 2560, 9216, mtp.v_mlp_act, lyr.down, mtp.mtp_h, true)?;

        // 4. mtp.norm on LAST row (m - 1) -> mtp_h_norm[0] -> lm_head -> argmax -> mtp_out_tok
        let mut last_h = mtp.mtp_h + ((m - 1) * 2560 * 2) as u64;
        let mut mnw = mw.norm_w;
        let mut hnorm = mtp.mtp_h_norm;
        self.cuda.launch(
            self.funcs.rmsnorm_2d,
            (1, 1, 1),
            (256, 1, 1),
            &mut [
                &mut last_h as *mut _ as *mut c_void,
                &mut mnw as *mut _ as *mut c_void,
                &mut hnorm as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;
        self.cuda.gemm_bf16_nt(2, self.cfg.vocab_size, 2560, mtp.mtp_h_norm, self.weights.lm_head_w, mtp.v_logits)?;

        let mut lg_p = mtp.v_logits;
        let mut av_p = mtp.v_arg_vals;
        let mut ai_p = mtp.v_arg_idxs;
        let mut mout = mtp.mtp_out_tok;
        self.cuda.launch(
            self.funcs.argmax_stage1,
            (256, 1, 1),
            (256, 1, 1),
            &mut [
                &mut lg_p as *mut _ as *mut c_void,
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut vocab_i as *mut _ as *mut c_void,
            ],
        )?;
        self.cuda.launch(
            self.funcs.argmax_stage2,
            (1, 1, 1),
            (256, 1, 1),
            &mut [
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut mout as *mut _ as *mut c_void,
                &mut nblk as *mut _ as *mut c_void,
            ],
        )?;
        Ok(())
    }

    fn run_mtp_step_small(
        &self,
        m: usize,
        start_pos: usize,
        tok_ids_gpu: CUdeviceptr,
        hid_gpu: CUdeviceptr,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().unwrap();
        let pos_i32 = start_pos as i32;
        self.cuda.htod_async(mtp.mtp_pos, &pos_i32 as *const i32 as *const u8, 4)?;
        self.run_mtp_step_small_body(m, tok_ids_gpu, hid_gpu)
    }

    /// Prefill 1-layer MTP on `seq_len` positions starting at `start_pos` using post-norm `h_norm [seq_len, 2560]`
    /// and `shifted_ids [seq_len]` -> populates MTP KV cache & optionally returns draft token from last row!
    fn run_mtp_prefill_range(
        &mut self,
        start_pos: usize,
        shifted_ids: &[i64],
        h_norm_gpu: CUdeviceptr,
        predict_draft: bool,
    ) -> Result<i64> {
        let seq_len = shifted_ids.len();
        if seq_len == 0 {
            return Ok(0);
        }
        let (p_ids, p_h, p_nrm, p_del, p_co, _p_dn_raw, p_big, p_med) = self.ensure_prefill_workspace(seq_len)?;
        // Note: h_norm_gpu might alias p_nrm! Copy shifted_ids -> p_ids, concat -> p_big [seq_len, 5120],
        // fc -> p_h [seq_len, 2560] (doesn't overwrite p_nrm until after concat!).
        self.cuda.htod_sync(p_ids, shifted_ids.as_ptr() as *const u8, seq_len * 8)?;

        let mtp = self.mtp.as_ref().unwrap();
        let mw = mtp.weights;
        let lyr = mw.layer;
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut seq_i: i32 = seq_len as i32;
        let mut start_i: i32 = start_pos as i32;
        let mut max_seq_i: i32 = self.max_seq as i32;
        let mut inter: i32 = 9216;
        let mut total_h: i32 = (seq_len * 2560) as i32;

        // 1. Fused embed + pre_fc_norm_emb + pre_fc_norm_hid + concat -> p_big [seq_len, 5120]
        let mut tid_p = p_ids;
        let mut emb_p = self.weights.embed_w;
        let mut hin_p = h_norm_gpu;
        let mut w_en = mw.pre_fc_norm_emb;
        let mut w_hn = mw.pre_fc_norm_hid;
        let mut cat_p = p_big;
        self.cuda.launch(
            self.funcs.mtp_pre_fc_concat,
            (seq_len as u32, 1, 1),
            (256, 1, 1),
            &mut [
                &mut tid_p as *mut _ as *mut c_void,
                &mut emb_p as *mut _ as *mut c_void,
                &mut hin_p as *mut _ as *mut c_void,
                &mut w_en as *mut _ as *mut c_void,
                &mut w_hn as *mut _ as *mut c_void,
                &mut cat_p as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;

        // 2. mtp.fc: [seq_len, 5120] -> p_h [seq_len, 2560]
        self.cuda.gemm_bf16_nt(seq_len.max(2), 2560, 5120, p_big, mw.fc_w, p_h)?;

        // 3. mtp.layers.0 (Full Attention + MLP)
        let mut h_p = p_h;
        let mut nw_p = lyr.in_norm;
        let mut nrm_p = p_nrm;
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

        self.cuda.gemm_bf16_nt(seq_len.max(2), 10240, 2560, p_nrm, lyr.qkv, p_big)?;
        let mut qkv_p = p_big;
        let mut qnw = lyr.q_norm;
        let mut knw = lyr.k_norm;
        let mut cos_p = self.cos_table;
        let mut sin_p = self.sin_table;
        let mut q_sdpa = p_med;
        let mut kc = mtp.mtp_k_cache;
        let mut vc = mtp.mtp_v_cache;
        self.cuda.launch(
            self.funcs.qk_rope_kv_prefill,
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
                &mut kc as *mut _ as *mut c_void,
                &mut vc as *mut _ as *mut c_void,
                &mut max_seq_i as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;
        let mut co = p_co;
        let q_tiles = ((seq_len + 1) / 2) as u32;
        self.cuda.launch(
            self.funcs.prefill_gqa_causal_gated,
            (4, q_tiles, 1),
            (256, 1, 1),
            &mut [
                &mut q_sdpa as *mut _ as *mut c_void,
                &mut qkv_p as *mut _ as *mut c_void,
                &mut kc as *mut _ as *mut c_void,
                &mut vc as *mut _ as *mut c_void,
                &mut start_i as *mut _ as *mut c_void,
                &mut seq_i as *mut _ as *mut c_void,
                &mut max_seq_i as *mut _ as *mut c_void,
                &mut co as *mut _ as *mut c_void,
            ],
        )?;
        self.cuda.gemm_bf16_nt(seq_len.max(2), 2560, 4096, p_co, lyr.o_proj, p_del)?;

        let mut del_p = p_del;
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

        self.cuda.gemm_bf16_nt(seq_len.max(2), 18432, 2560, p_nrm, lyr.gate_up, p_big)?;
        let mut gu_p = p_big;
        let mut act_p = p_med;
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
        self.cuda.gemm_bf16_nt(seq_len.max(2), 2560, 9216, p_med, lyr.down, p_del)?;

        let blocks = ((seq_len * 2560 + 255) / 256) as u32;
        self.cuda.launch(
            self.funcs.residual_add,
            (blocks, 1, 1),
            (256, 1, 1),
            &mut [
                &mut h_p as *mut _ as *mut c_void,
                &mut del_p as *mut _ as *mut c_void,
                &mut total_h as *mut _ as *mut c_void,
            ],
        )?;

        if !predict_draft {
            return Ok(0);
        }

        // Last row -> mtp.norm -> mtp.mtp_h_norm -> lm_head -> argmax -> draft_1!
        let mut last_h = p_h + ((seq_len - 1) * 2560 * 2) as u64;
        let mut mnw = mw.norm_w;
        let mut hnorm = mtp.mtp_h_norm;
        let mut vocab_i: i32 = self.cfg.vocab_size as i32;
        let mut nblk: i32 = 256;
        self.cuda.launch(
            self.funcs.rmsnorm_2d,
            (1, 1, 1),
            (256, 1, 1),
            &mut [
                &mut last_h as *mut _ as *mut c_void,
                &mut mnw as *mut _ as *mut c_void,
                &mut hnorm as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
                &mut eps as *mut _ as *mut c_void,
            ],
        )?;
        self.cuda.gemm_bf16_nt(2, self.cfg.vocab_size, 2560, mtp.mtp_h_norm, self.weights.lm_head_w, mtp.v_logits)?;
        let mut lg_p = mtp.v_logits;
        let mut av_p = mtp.v_arg_vals;
        let mut ai_p = mtp.v_arg_idxs;
        let mut mout = mtp.mtp_out_tok;
        self.cuda.launch(
            self.funcs.argmax_stage1,
            (256, 1, 1),
            (256, 1, 1),
            &mut [
                &mut lg_p as *mut _ as *mut c_void,
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut vocab_i as *mut _ as *mut c_void,
            ],
        )?;
        self.cuda.launch(
            self.funcs.argmax_stage2,
            (1, 1, 1),
            (256, 1, 1),
            &mut [
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut mout as *mut _ as *mut c_void,
                &mut nblk as *mut _ as *mut c_void,
            ],
        )?;
        Ok(0)
    }

    /// Phase R7.1: Full-Cycle CUDA Graph Launch for MTP Draft Chain (d2..dK) + Parallel Verify M=4!
    /// Replaces ~45 eager launches + verify_graph with ONE SINGLE CUDA GRAPH LAUNCH (`draft_and_verify_graph`)!
    fn enqueue_mtp_draft_and_verify(&self, cur_tok: i64, cur_pos: usize) -> Result<()> {
        let mtp = self.mtp.as_ref().unwrap();
        let pos_i32 = cur_pos as i32;
        self.cuda.htod_async(mtp.v_tok_ids, &cur_tok as *const i64 as *const u8, 8)?;
        self.cuda.htod_async(mtp.mtp_pos, &pos_i32 as *const i32 as *const u8, 4)?;
        self.cuda.htod_async(mtp.v_pos, &pos_i32 as *const i32 as *const u8, 4)?;
        self.cuda.launch_graph(mtp.draft_and_verify_graph)?;
        Ok(())
    }

    /// Phase R7.1: Full-Cycle CUDA Graph Launch for Commit DeltaNet + MTP d1 + d2..dK + Next Verify M=4!
    /// ONLY 2 CUDA GRAPH LAUNCHES for the ENTIRE MTP K=3 cycle (ZERO eager kernel launches)!
    fn enqueue_mtp_commit_and_next_verify(
        &self,
        accepted: usize,
        v_toks: &[i64; 4],
        cur_pos: usize,
    ) -> Result<()> {
        let mtp = self.mtp.as_ref().unwrap();
        let num_new = accepted + 1;
        let old_pos_i32 = cur_pos as i32;
        let new_pos_i32 = (cur_pos + num_new) as i32;
        let next_tok = v_toks[accepted];

        // Graph 1: Commit DeltaNet (conv + rec) + MTP d1 (reading old_pos_i32)
        self.cuda.htod_async(mtp.mtp_tok_ids, v_toks.as_ptr() as *const u8, num_new * 8)?;
        self.cuda.htod_async(mtp.mtp_pos, &old_pos_i32 as *const i32 as *const u8, 4)?;
        self.cuda.launch_graph(mtp.commit_and_d1_graphs[accepted])?;

        // Graph 2: MTP d2..dK + Verify M=4 + async D2H of drafts & verified toks (reading new_pos_i32)
        self.cuda.htod_async(mtp.v_tok_ids, &next_tok as *const i64 as *const u8, 8)?;
        self.cuda.htod_async(mtp.mtp_pos, &new_pos_i32 as *const i32 as *const u8, 4)?;
        self.cuda.htod_async(mtp.v_pos, &new_pos_i32 as *const i32 as *const u8, 4)?;
        self.cuda.launch_graph(mtp.draft_and_verify_graph)?;
        Ok(())
    }

    fn acquire_ram_buf(&mut self, needed: usize) -> Vec<u8> {
        let mut buf = if let Some(b) = self.free_ram_pool.pop() {
            b
        } else if self.prefix_lru.len() >= self.prefix_cache_max_slots {
            if let Some(old_k) = self.prefix_lru.pop_front() {
                self.prefix_cache.remove(&old_k).map(|s| s.data).unwrap_or_default()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if buf.capacity() < needed {
            buf.reserve(needed - buf.len());
        }
        unsafe { buf.set_len(needed); } // ZERO memset(0) — overwritten immediately by DMA!
        buf
    }

    /// Snapshot GPU caches through the 16 MiB batched pinned staging buffer into a recycled Host RAM Vec<u8>.
    /// Packs multiple tensors per flush (~4 syncs total instead of 121!) -> ~4 ms full snapshot, 0 MiB WDDM VRAM bloat!
    fn capture_pinned_snapshot(&mut self, seq_len: usize, last_h_gpu: CUdeviceptr, use_rb_snap: bool) -> Result<PinnedSnapshot> {
        let p1_b = self.p1_cache.calc_snapshot_bytes(seq_len);
        let p2_b = self.p2_cache.calc_snapshot_bytes(seq_len);
        let has_mtp = self.mtp.is_some() && seq_len > 1;
        let mtp_kv_b = if has_mtp { 4 * (seq_len - 1) * 256 * 2 * 2 } else { 0 };
        let lh_b = if has_mtp { 2560 * 2 } else { 0 };
        let total = p1_b + p2_b + mtp_kv_b + lh_b;

        let mut data = self.acquire_ram_buf(total);
        let conv_bytes = 8192 * 4 * 2;
        let rec_bytes = 32 * 128 * 128 * 4;
        let kv_head_bytes = seq_len * 256 * 2;
        let kv_stride = self.max_seq * 256 * 2;

        let stg_ptr = self.pinned_staging.as_ref().unwrap().ptr;
        let stg_cap = self.pinned_staging.as_ref().unwrap().cap;
        let cuda = &self.cuda;

        let mut batch_used = 0usize;
        let mut batch_dst_base = 0usize;
        let mut push_dtoh = |src_gpu: CUdeviceptr, bytes: usize| -> Result<()> {
            let mut done = 0usize;
            while done < bytes {
                if batch_used == stg_cap {
                    cuda.sync()?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(stg_ptr, data.as_mut_ptr().add(batch_dst_base), batch_used);
                    }
                    batch_dst_base += batch_used;
                    batch_used = 0;
                }
                let avail = stg_cap - batch_used;
                let chunk = (bytes - done).min(avail);
                cuda.dtoh_async(unsafe { stg_ptr.add(batch_used) }, src_gpu + done as u64, chunk)?;
                batch_used += chunk;
                done += chunk;
            }
            Ok(())
        };

        for stage in 0..2 {
            let layers = if stage == 0 { &self.p1_cache.layers } else { &self.p2_cache.layers };
            let rbs = if use_rb_snap {
                let m = self.mtp.as_ref().unwrap();
                Some(if stage == 0 { m.p1_rb.as_slice() } else { m.p2_rb.as_slice() })
            } else {
                None
            };
            for (i, l) in layers.iter().enumerate() {
                if l.is_linear {
                    let (cs, rs) = if let Some(rb) = rbs {
                        (rb[i].conv_snaps, rb[i].rec_final)
                    } else {
                        (l.conv_state, l.rec_state)
                    };
                    push_dtoh(cs, conv_bytes)?;
                    push_dtoh(rs, rec_bytes)?;
                } else {
                    for h in 0..4 {
                        push_dtoh(l.k_cache + (h * kv_stride) as u64, kv_head_bytes)?;
                    }
                    for h in 0..4 {
                        push_dtoh(l.v_cache + (h * kv_stride) as u64, kv_head_bytes)?;
                    }
                }
            }
        }

        let last_h_off = total - lh_b;
        if has_mtp {
            let mtp = self.mtp.as_ref().unwrap();
            let mlen = seq_len - 1;
            let mkv_b = mlen * 256 * 2;
            for h in 0..4 {
                push_dtoh(mtp.mtp_k_cache + (h * kv_stride) as u64, mkv_b)?;
            }
            for h in 0..4 {
                push_dtoh(mtp.mtp_v_cache + (h * kv_stride) as u64, mkv_b)?;
            }
            push_dtoh(last_h_gpu, 2560 * 2)?;
        }
        if batch_used > 0 {
            cuda.sync()?;
            unsafe {
                std::ptr::copy_nonoverlapping(stg_ptr, data.as_mut_ptr().add(batch_dst_base), batch_used);
            }
        }
        Ok(PinnedSnapshot { seq_len, has_mtp, data, last_h_off })
    }

    /// Restore a snapshot from Host RAM Vec<u8> through the 16 MiB batched pinned staging buffer -> GPU (~4 syncs!).
    fn restore_pinned_snapshot(&mut self, key: &Vec<i64>) -> Result<bool> {
        let snap = self.prefix_cache.remove(key).unwrap();
        let seq_len = snap.seq_len;
        let has_mtp = snap.has_mtp;
        let conv_bytes = 8192 * 4 * 2;
        let rec_bytes = 32 * 128 * 128 * 4;
        let kv_head_bytes = seq_len * 256 * 2;
        let kv_stride = self.max_seq * 256 * 2;

        let stg_ptr = self.pinned_staging.as_ref().unwrap().ptr;
        let stg_cap = self.pinned_staging.as_ref().unwrap().cap;
        let cuda = &self.cuda;

        let mut batch_used = 0usize;
        let mut src_off = 0usize;
        let mut push_htod = |dst_gpu: CUdeviceptr, bytes: usize| -> Result<()> {
            let mut done = 0usize;
            while done < bytes {
                if batch_used == stg_cap {
                    cuda.sync()?;
                    batch_used = 0;
                }
                let avail = stg_cap - batch_used;
                let chunk = (bytes - done).min(avail);
                unsafe {
                    std::ptr::copy_nonoverlapping(snap.data.as_ptr().add(src_off), stg_ptr.add(batch_used), chunk);
                }
                cuda.htod_async(dst_gpu + done as u64, unsafe { stg_ptr.add(batch_used) }, chunk)?;
                batch_used += chunk;
                src_off += chunk;
                done += chunk;
            }
            Ok(())
        };

        for stage in 0..2 {
            let layers = if stage == 0 { &self.p1_cache.layers } else { &self.p2_cache.layers };
            for l in layers {
                if l.is_linear {
                    push_htod(l.conv_state, conv_bytes)?;
                    push_htod(l.rec_state, rec_bytes)?;
                } else {
                    for h in 0..4 {
                        push_htod(l.k_cache + (h * kv_stride) as u64, kv_head_bytes)?;
                    }
                    for h in 0..4 {
                        push_htod(l.v_cache + (h * kv_stride) as u64, kv_head_bytes)?;
                    }
                }
            }
        }

        let mut mtp_ok = false;
        if has_mtp && self.mtp.is_some() && seq_len > 1 {
            let mtp = self.mtp.as_ref().unwrap();
            let mlen = seq_len - 1;
            let mkv_b = mlen * 256 * 2;
            for h in 0..4 {
                push_htod(mtp.mtp_k_cache + (h * kv_stride) as u64, mkv_b)?;
            }
            for h in 0..4 {
                push_htod(mtp.mtp_v_cache + (h * kv_stride) as u64, mkv_b)?;
            }
            push_htod(self.s_live_last_h, 2560 * 2)?;
            mtp_ok = true;
        }
        if batch_used > 0 {
            cuda.sync()?;
        }
        self.prefix_cache.insert(key.clone(), snap);
        Ok(mtp_ok)
    }

    fn insert_prefix_snapshot(&mut self, key: Vec<i64>, snap: PinnedSnapshot) {
        if self.prefix_cache_max_slots == 0 {
            return;
        }
        if let Some(old_s) = self.prefix_cache.remove(&key) {
            self.prefix_lru.retain(|k| k != &key);
            self.free_ram_pool.push(old_s.data);
        }
        self.prefix_cache.insert(key.clone(), snap);
        self.prefix_lru.push_back(key);
        while self.prefix_lru.len() > self.prefix_cache_max_slots {
            if let Some(old) = self.prefix_lru.pop_front() {
                if let Some(old_s) = self.prefix_cache.remove(&old) {
                    self.free_ram_pool.push(old_s.data);
                }
            }
        }
    }

    fn ensure_prefill_workspace(&mut self, seq_len: usize) -> Result<(CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr)> {
        // Keep pf_arena compact (<= ~47 MiB at chunk=512) so --max-seq 4096 never blows RTX 3080 10 GB VRAM!
        let cap = if seq_len <= 128 {
            128
        } else if seq_len <= 256 {
            256
        } else if seq_len <= 512 {
            512
        } else {
            seq_len.min(self.max_seq)
        };
        if cap > self.pf_cap_seq {
            if self.pf_arena != 0 {
                self.cuda.free(self.pf_arena);
            }
            let align = |b: usize| (b + 255) & !255;
            let bytes = align(cap * 8)
                + align(cap * 2560 * 2) * 3
                + align(cap * 4096 * 2)
                + align(cap * 4096 * 4)
                + align(cap * 18432 * 2)
                + align(cap * 9216 * 2);
            self.pf_arena = self.cuda.alloc(bytes)?;
            self.pf_cap_seq = cap;
        }
        let align = |b: usize| (b + 255) & !255;
        let c = self.pf_cap_seq;
        let mut off = 0u64;
        let mut take = |b: usize| {
            let p = self.pf_arena + off;
            off += align(b) as u64;
            p
        };
        let p_ids = take(c * 8);
        let p_h = take(c * 2560 * 2);
        let p_nrm = take(c * 2560 * 2);
        let p_del = take(c * 2560 * 2);
        let p_co = take(c * 4096 * 2);
        let p_dn_raw = take(c * 4096 * 4);
        let p_big = take(c * 18432 * 2);
        let p_med = take(c * 9216 * 2);
        Ok((p_ids, p_h, p_nrm, p_del, p_co, p_dn_raw, p_big, p_med))
    }

    fn run_stage_prefill_ops(
        &self,
        layers: &[PackedLayerPtrs],
        caches: &[LayerCachePtrs],
        rbs: Option<&[LinearRollbackPtrs]>,
        h: CUdeviceptr,
        normed: CUdeviceptr,
        delta: CUdeviceptr,
        core_out: CUdeviceptr,
        dn_raw: CUdeviceptr,
        ws_big: CUdeviceptr,
        ws_med: CUdeviceptr,
        start_pos: usize,
        seq_len: usize,
        use_init_state: i32,
        snap_t: i32,
    ) -> Result<()> {
        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut seq_i: i32 = seq_len as i32;
        let mut start_i: i32 = start_pos as i32;
        let mut max_seq_i: i32 = self.max_seq as i32;
        let mut c8192: i32 = 8192;
        let mut stride_dn: i32 = 12352;
        let mut inter: i32 = 9216;
        let mut total_h: i32 = (seq_len * 2560) as i32;
        let mut use_init = use_init_state;
        let mut snap_i = snap_t;

        for (idx, (lyr, lc)) in layers.iter().zip(caches.iter()).enumerate() {
            let mut h_p = h;
            let mut nw_p = lyr.in_norm;
            let mut nrm_p = normed;
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
                self.cuda.gemm_bf16_nt(seq_len, 12352, 2560, normed, lyr.in_all, ws_big)?;
                let mut pin = ws_big;
                let mut cs = lc.conv_state;
                let mut cs_snap = rbs.map(|r| r[idx].conv_snaps).unwrap_or(0);
                let mut cw = lyr.conv_w;
                let mut cv_out = ws_med;
                if seq_len >= 64 {
                    let t_tiles = ((seq_len + 31) / 32) as u32;
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
                            &mut seq_i as *mut _ as *mut c_void,
                            &mut c8192 as *mut _ as *mut c_void,
                            &mut stride_dn as *mut _ as *mut c_void,
                            &mut use_init as *mut _ as *mut c_void,
                            &mut snap_i as *mut _ as *mut c_void,
                        ],
                    )?;
                    self.cuda.launch(
                        self.funcs.conv1d_update_state,
                        (32, 1, 1),
                        (256, 1, 1),
                        &mut [
                            &mut pin as *mut _ as *mut c_void,
                            &mut cs as *mut _ as *mut c_void,
                            &mut seq_i as *mut _ as *mut c_void,
                            &mut c8192 as *mut _ as *mut c_void,
                            &mut stride_dn as *mut _ as *mut c_void,
                            &mut use_init as *mut _ as *mut c_void,
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
                            &mut seq_i as *mut _ as *mut c_void,
                            &mut c8192 as *mut _ as *mut c_void,
                            &mut stride_dn as *mut _ as *mut c_void,
                            &mut use_init as *mut _ as *mut c_void,
                            &mut snap_i as *mut _ as *mut c_void,
                        ],
                    )?;
                }

                let rec_snap = rbs.map(|r| r[idx].rec_final).unwrap_or(0);
                // Hybrid crossover verified via nsys: sequential wins for T < 192 (1 launch),
                // WMMA Chunked WY Tensor Core wins decisively for T >= 192 (1.80x faster at T=512!)
                if seq_len >= 192 && seq_len <= 513 {
                    run_deltanet_prefill_wy(
                        &self.cuda,
                        &self.funcs.wy,
                        cv_out,
                        pin,
                        lyr.dt_bias,
                        lyr.a_log_f32,
                        lyr.dn_norm,
                        lc.rec_state,
                        rec_snap,
                        dn_raw,
                        ws_big,
                        core_out,
                        seq_len,
                        use_init,
                        snap_i,
                        eps,
                    )?;
                } else {
                    let mut dtb = lyr.dt_bias;
                    let mut alog = lyr.a_log_f32;
                    let mut dnw = lyr.dn_norm;
                    let mut rec = lc.rec_state;
                    let mut rec_snap_p = rec_snap;
                    let mut raw_p = dn_raw;
                    let mut co = core_out;
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
                            &mut rec_snap_p as *mut _ as *mut c_void,
                            &mut raw_p as *mut _ as *mut c_void,
                            &mut seq_i as *mut _ as *mut c_void,
                            &mut use_init as *mut _ as *mut c_void,
                            &mut snap_i as *mut _ as *mut c_void,
                        ],
                    )?;
                    self.cuda.launch(
                        self.funcs.deltanet_prefill_norm_gate,
                        (32, seq_len as u32, 1),
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
                self.cuda.gemm_bf16_nt(seq_len, 2560, 4096, core_out, lyr.out_proj, delta)?;
            } else {
                self.cuda.gemm_bf16_nt(seq_len, 10240, 2560, normed, lyr.qkv, ws_big)?;
                let mut qkv_p = ws_big;
                let mut qnw = lyr.q_norm;
                let mut knw = lyr.k_norm;
                let mut cos_p = self.cos_table;
                let mut sin_p = self.sin_table;
                let mut q_sdpa = ws_med;
                let mut kc = lc.k_cache;
                let mut vc = lc.v_cache;
                self.cuda.launch(
                    self.funcs.qk_rope_kv_prefill,
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
                        &mut kc as *mut _ as *mut c_void,
                        &mut vc as *mut _ as *mut c_void,
                        &mut max_seq_i as *mut _ as *mut c_void,
                        &mut eps as *mut _ as *mut c_void,
                    ],
                )?;
                let mut co = core_out;
                if seq_len >= 64 {
                    let q_tiles = ((seq_len + 15) / 16) as u32;
                    self.cuda.launch(
                        self.funcs.prefill_gqa_wmma,
                        (16, q_tiles, 1),
                        (128, 1, 1),
                        &mut [
                            &mut q_sdpa as *mut _ as *mut c_void,
                            &mut qkv_p as *mut _ as *mut c_void,
                            &mut kc as *mut _ as *mut c_void,
                            &mut vc as *mut _ as *mut c_void,
                            &mut start_i as *mut _ as *mut c_void,
                            &mut seq_i as *mut _ as *mut c_void,
                            &mut max_seq_i as *mut _ as *mut c_void,
                            &mut co as *mut _ as *mut c_void,
                        ],
                    )?;
                } else {
                    let q_tiles = ((seq_len + 1) / 2) as u32;
                    self.cuda.launch(
                        self.funcs.prefill_gqa_causal_gated,
                        (4, q_tiles, 1),
                        (256, 1, 1),
                        &mut [
                            &mut q_sdpa as *mut _ as *mut c_void,
                            &mut qkv_p as *mut _ as *mut c_void,
                            &mut kc as *mut _ as *mut c_void,
                            &mut vc as *mut _ as *mut c_void,
                            &mut start_i as *mut _ as *mut c_void,
                            &mut seq_i as *mut _ as *mut c_void,
                            &mut max_seq_i as *mut _ as *mut c_void,
                            &mut co as *mut _ as *mut c_void,
                        ],
                    )?;
                }
                self.cuda.gemm_bf16_nt(seq_len, 2560, 4096, core_out, lyr.o_proj, delta)?;
            }

            let mut del_p = delta;
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

            self.cuda.gemm_bf16_nt(seq_len, 18432, 2560, normed, lyr.gate_up, ws_big)?;
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
            self.cuda.gemm_bf16_nt(seq_len, 2560, 9216, ws_med, lyr.down, delta)?;

            let blocks = ((seq_len * 2560 + 255) / 256) as u32;
            self.cuda.launch(
                self.funcs.residual_add,
                (blocks, 1, 1),
                (256, 1, 1),
                &mut [
                    &mut h_p as *mut _ as *mut c_void,
                    &mut del_p as *mut _ as *mut c_void,
                    &mut total_h as *mut _ as *mut c_void,
                ],
            )?;
        }
        Ok(())
    }

    fn prefill_segment_main(
        &mut self,
        seg: &[i64],
        start_pos: usize,
        compute_all_norm: bool,
        nrm_row_off: usize,
        snap_t: i32,
    ) -> Result<(CUdeviceptr, CUdeviceptr)> {
        let seq_len = seg.len();
        let (p_ids, p_h, p_nrm, p_del, p_co, p_dn_raw, p_big, p_med) = self.ensure_prefill_workspace(seq_len + nrm_row_off)?;
        self.cuda.htod_async(p_ids, seg.as_ptr() as *const u8, seq_len * 8)?;

        let mut ids_p = p_ids;
        let mut emb_p = self.weights.embed_w;
        let mut h_p = p_h;
        let mut sl: i32 = seq_len as i32;
        let mut dim: i32 = 2560;
        self.cuda.launch(
            self.funcs.embed_prefill,
            (10, seq_len as u32, 1),
            (256, 1, 1),
            &mut [
                &mut ids_p as *mut _ as *mut c_void,
                &mut emb_p as *mut _ as *mut c_void,
                &mut h_p as *mut _ as *mut c_void,
                &mut sl as *mut _ as *mut c_void,
                &mut dim as *mut _ as *mut c_void,
            ],
        )?;

        let use_init = if start_pos > 0 { 1 } else { 0 };
        let (p1_rb, p2_rb) = if snap_t > 0 {
            let m = self.mtp.as_ref().unwrap();
            (Some(m.p1_rb.as_slice()), Some(m.p2_rb.as_slice()))
        } else {
            (None, None)
        };
        self.run_stage_prefill_ops(&self.weights.p1_layers, &self.p1_cache.layers, p1_rb, p_h, p_nrm, p_del, p_co, p_dn_raw, p_big, p_med, start_pos, seq_len, use_init, snap_t)?;
        self.run_stage_prefill_ops(&self.weights.p2_layers, &self.p2_cache.layers, p2_rb, p_h, p_nrm, p_del, p_co, p_dn_raw, p_big, p_med, start_pos, seq_len, use_init, snap_t)?;

        if compute_all_norm {
            let mut fnw = self.weights.final_norm_w;
            let mut eps: f32 = self.cfg.rms_norm_eps;
            let nrm_dst = p_nrm + (nrm_row_off * 2560 * 2) as u64;
            let mut nrm_p = nrm_dst;
            self.cuda.launch(
                self.funcs.rmsnorm_2d,
                (seq_len as u32, 1, 1),
                (256, 1, 1),
                &mut [
                    &mut h_p as *mut _ as *mut c_void,
                    &mut fnw as *mut _ as *mut c_void,
                    &mut nrm_p as *mut _ as *mut c_void,
                    &mut dim as *mut _ as *mut c_void,
                    &mut eps as *mut _ as *mut c_void,
                ],
            )?;
        }
        Ok((p_h, p_nrm))
    }

    pub fn prefill_fused(
        &mut self,
        prompt_ids: &[i64],
        sys_prefix_len: usize,
        hist_prefix_len: usize,
        use_prefix_cache: bool,
    ) -> Result<PrefillResult> {
        self.cuda.bind_context()?;
        let mut cached_len = 0usize;
        let mut cache_tier: &'static str = "miss";
        let mut cache_restore_ms: f64 = 0.0;
        let mut has_prev_h_norm = false;

        let enable_cache = use_prefix_cache && self.prefix_cache_max_slots > 0;
        if enable_cache {
            // Check Tier 0: Zero-Copy Live GPU Hot-Cache (0.0 ms restore!)
            let t0_len = if !self.live_tokens.is_empty()
                && self.live_tokens.len() < prompt_ids.len()
                && &prompt_ids[..self.live_tokens.len()] == self.live_tokens.as_slice()
                && (self.mtp.is_none() || self.live_has_mtp)
            {
                self.live_tokens.len()
            } else {
                0
            };

            // Check Tier 1: Host-RAM (Pinned-DMA) Longest Prefix Match
            let mut best_key: Option<Vec<i64>> = None;
            let mut t1_len = 0usize;
            for k in self.prefix_cache.keys() {
                if k.len() < prompt_ids.len() && k.len() > t1_len && &prompt_ids[..k.len()] == k.as_slice() {
                    let e = self.prefix_cache.get(k).unwrap();
                    if self.mtp.is_none() || e.has_mtp {
                        t1_len = k.len();
                        best_key = Some(k.clone());
                    }
                }
            }

            if t0_len > 0 && t0_len + 16 >= t1_len {
                cached_len = t0_len;
                cache_tier = "gpu_hot";
                cache_restore_ms = 0.0;
                has_prev_h_norm = self.mtp.is_some();
                self.prefix_cache_hits += 1;
                self.prefix_cache_saved_tokens += cached_len as u64;
            } else if let Some(key) = best_key {
                let t_res = Instant::now();
                let mtp_ok = self.restore_pinned_snapshot(&key)?;
                self.cuda.sync()?;
                cache_restore_ms = t_res.elapsed().as_secs_f64() * 1000.0;
                self.prefix_lru.retain(|k| k != &key);
                self.prefix_lru.push_back(key);

                cached_len = t1_len;
                cache_tier = "ram_pinned";
                has_prev_h_norm = mtp_ok;
                self.prefix_cache_hits += 1;
                self.prefix_cache_saved_tokens += cached_len as u64;
            }
        }

        const PREFILL_CHUNK: usize = 512;
        let mtp_enabled = self.mtp.is_some();
        let mut cur_pos = cached_len;
        let mut sys_cached_in_chunk = false;

        // Phase R5: 512-Token Chunked Prefill (keeps pf_arena <= 46.6 MiB even at --max-seq 4096!)
        // For prompts <= 512 tokens (or prefix-cached suffixes <= 512), this loop runs 0 times -> 100% single-pass!
        while prompt_ids.len() - cur_pos > PREFILL_CHUNK {
            let seg_len = PREFILL_CHUNK;
            let seg = &prompt_ids[cur_pos..cur_pos + seg_len];
            let row_off = if mtp_enabled && has_prev_h_norm { 1 } else { 0 };
            let sys_in_this_chunk = enable_cache
                && mtp_enabled
                && !sys_cached_in_chunk
                && sys_prefix_len >= 12
                && sys_prefix_len > cur_pos
                && sys_prefix_len <= cur_pos + seg_len;
            let chunk_snap_t: i32 = if sys_in_this_chunk {
                (sys_prefix_len - cur_pos) as i32
            } else {
                0
            };

            let (_, p_nrm) = self.prefill_segment_main(seg, cur_pos, mtp_enabled, row_off, chunk_snap_t)?;
            if mtp_enabled {
                if has_prev_h_norm {
                    self.cuda.dtod_async(p_nrm, self.s_live_last_h, 2560 * 2)?;
                }
                if sys_in_this_chunk {
                    let sys_h_ptr = p_nrm + ((row_off + (sys_prefix_len - cur_pos) - 1) * 2560 * 2) as u64;
                    self.cuda.dtod_async(self.s_normed, sys_h_ptr, 2560 * 2)?;
                }
                let last_ptr = p_nrm + ((row_off + seg_len - 1) * 2560 * 2) as u64;
                self.cuda.dtod_async(self.s_live_last_h, last_ptr, 2560 * 2)?;
                if has_prev_h_norm {
                    let _ = self.run_mtp_prefill_range(cur_pos - 1, seg, p_nrm, false)?;
                } else if seg_len > 1 {
                    let _ = self.run_mtp_prefill_range(0, &prompt_ids[1..seg_len], p_nrm, false)?;
                }
                if sys_in_this_chunk {
                    let snap = self.capture_pinned_snapshot(sys_prefix_len, self.s_normed, true)?;
                    self.insert_prefix_snapshot(prompt_ids[..sys_prefix_len].to_vec(), snap);
                    sys_cached_in_chunk = true;
                }
                has_prev_h_norm = true;
            }
            cur_pos += seg_len;
        }

        // Final Pass over remaining suffix `prompt_ids[cur_pos ..]` (<= 512 tokens, 0.0 ms snapshot overhead on TTFT!).
        let suffix = &prompt_ids[cur_pos..];
        let seq_len = suffix.len();
        let deferred_hist_snap = if enable_cache && mtp_enabled {
            if !sys_cached_in_chunk && sys_prefix_len >= 12 && sys_prefix_len > cur_pos && sys_prefix_len < prompt_ids.len() {
                sys_prefix_len
            } else if hist_prefix_len >= 24 && hist_prefix_len > cur_pos && hist_prefix_len < prompt_ids.len() {
                hist_prefix_len
            } else {
                0
            }
        } else {
            0
        };
        let snap_t: i32 = if deferred_hist_snap > 0 {
            (deferred_hist_snap - cur_pos) as i32
        } else {
            0
        };

        let row_off = if mtp_enabled && has_prev_h_norm { 1 } else { 0 };
        let (p_h, p_nrm) = self.prefill_segment_main(suffix, cur_pos, mtp_enabled, row_off, snap_t)?;

        let mut dim: i32 = 2560;
        let mut eps: f32 = self.cfg.rms_norm_eps;
        let mut fnw = self.weights.final_norm_w;
        if mtp_enabled {
            if has_prev_h_norm {
                self.cuda.dtod_async(p_nrm, self.s_live_last_h, 2560 * 2)?;
            }
            let last_nrm = p_nrm + ((row_off + seq_len - 1) * 2560 * 2) as u64;
            self.cuda.dtod_async(self.s_normed, last_nrm, 2560 * 2)?;
            if deferred_hist_snap > 0 {
                // Stage hist_prefix_len's post-norm hidden state into s_live_last_h for deferred snapshot after first_tok!
                let hist_h_ptr = p_nrm + ((row_off + (deferred_hist_snap - cur_pos) - 1) * 2560 * 2) as u64;
                self.cuda.dtod_async(self.s_live_last_h, hist_h_ptr, 2560 * 2)?;
            } else {
                self.cuda.dtod_async(self.s_live_last_h, last_nrm, 2560 * 2)?;
            }
        } else {
            let last_row = p_h + ((seq_len - 1) * 2560 * 2) as u64;
            self.cuda.dtod_async(self.s_bridge, last_row, 2560 * 2)?;
            let mut br_p = self.s_bridge;
            let mut nrm = self.s_normed;
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
        }
        self.cuda.gemm_bf16_nt(2, self.cfg.vocab_size, 2560, self.s_normed, self.weights.lm_head_w, self.s_logits)?;

        let mut lg_p = self.s_logits;
        let mut av_p = self.s_arg_vals;
        let mut ai_p = self.s_arg_idxs;
        let mut vocab_i: i32 = self.cfg.vocab_size as i32;
        let mut nblk: i32 = 256;
        let mut ntok_p = self.s_next_token_id;
        self.cuda.launch(
            self.funcs.argmax_stage1,
            (256, 1, 1),
            (256, 1, 1),
            &mut [
                &mut lg_p as *mut _ as *mut c_void,
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut vocab_i as *mut _ as *mut c_void,
            ],
        )?;
        self.cuda.launch(
            self.funcs.argmax_stage2,
            (1, 1, 1),
            (256, 1, 1),
            &mut [
                &mut av_p as *mut _ as *mut c_void,
                &mut ai_p as *mut _ as *mut c_void,
                &mut ntok_p as *mut _ as *mut c_void,
                &mut nblk as *mut _ as *mut c_void,
            ],
        )?;

        self.cuda.sync()?;
        let mut first_tok: i64 = 0;
        self.cuda.dtoh_sync(&mut first_tok as *mut i64 as *mut u8, self.s_next_token_id, 8)?;

        self.live_tokens = prompt_ids.to_vec();
        self.live_has_mtp = false;

        Ok(PrefillResult {
            first_tok,
            cached_tokens: cached_len,
            cache_tier,
            cache_restore_ms,
            mtp_suffix_start: cur_pos,
            deferred_hist_snap,
        })
    }

    pub fn generate_stream<F>(
        &mut self,
        prompt_ids: &[i64],
        sys_prefix_len: usize,
        hist_prefix_len: usize,
        max_new_tokens: usize,
        ignore_eos: bool,
        mut on_token: F,
    ) -> Result<()>
    where
        F: FnMut(StreamToken) -> bool,
    {
        self.cuda.bind_context()?;
        let mtp_guard = if self.mtp.is_some() { 5 } else { 1 };
        let mut pids = prompt_ids;
        let mut spl = sys_prefix_len;
        let mut hpl = hist_prefix_len;
        let mut eff_max_new = max_new_tokens.max(1);

        // Phase R5: NEVER silently drop MTP K=3 to M=1 or discard Prefix Cache when nearing max_seq!
        if pids.len() + eff_max_new + mtp_guard > self.max_seq {
            if pids.len() + mtp_guard + 16 <= self.max_seq {
                // Keep the full prompt + Prefix Cache intact; clamp generation tokens to fit max_seq with MTP guard!
                eff_max_new = self.max_seq - pids.len() - mtp_guard;
            } else {
                // Prompt alone exceeds max_seq - 64: slide window while ALWAYS preserving MTP guard!
                let reserve_gen = eff_max_new.min(512);
                let keep = self.max_seq.saturating_sub(reserve_gen + mtp_guard).max(64);
                pids = &pids[pids.len().saturating_sub(keep)..];
                spl = 0;
                hpl = 0;
                eff_max_new = eff_max_new.min(self.max_seq.saturating_sub(pids.len() + mtp_guard).max(1));
            }
        }
        let max_new_tokens = eff_max_new;
        let prompt_len = pids.len();

        let pf = self.prefill_fused(pids, spl, hpl, true)?;
        let first_tok = pf.first_tok;
        let cached_tokens = pf.cached_tokens;
        let cache_tier = pf.cache_tier;
        let cache_restore_ms = pf.cache_restore_ms;
        let is_eos = self.tok.is_eos(first_tok) && !ignore_eos;

        if is_eos || max_new_tokens <= 1 {
            let _ = on_token(StreamToken {
                token_id: first_tok,
                text: if is_eos { String::new() } else { self.tok.decode_single(first_tok) },
                prompt_tokens: prompt_len,
                cached_tokens,
                cache_tier,
                cache_restore_ms,
                completion_tokens: 1,
                finish_reason: Some(if is_eos { "stop" } else { "length" }),
            });
            return Ok(());
        }

        // ---- Phase R3 + R4: Zero-Sync GPU-Resident MTP Speculative Decoding Path ----
        if self.mtp.is_some() && prompt_len + max_new_tokens + 4 < self.max_seq {
            // Yield first_tok IMMEDIATELY -> ZERO MTP overhead on TTFT!
            let keep_going = on_token(StreamToken {
                token_id: first_tok,
                text: self.tok.decode_single(first_tok),
                prompt_tokens: prompt_len,
                cached_tokens,
                cache_tier,
                cache_restore_ms,
                completion_tokens: 1,
                finish_reason: None,
            });
            if !keep_going {
                return Ok(());
            }

            // Populate MTP KV cache for suffix + get draft_1 on GPU (ZERO sync!)
            let (mtp_start_pos, shifted_ids): (usize, Vec<i64>) = if pf.mtp_suffix_start > 0 {
                let mut v = Vec::with_capacity(prompt_len - pf.mtp_suffix_start + 1);
                v.extend_from_slice(&pids[pf.mtp_suffix_start..]);
                v.push(first_tok);
                (pf.mtp_suffix_start - 1, v)
            } else {
                let mut v = Vec::with_capacity(prompt_len);
                v.extend_from_slice(&pids[1..]);
                v.push(first_tok);
                (0, v)
            };
            let (_, _, p_nrm, _, _, _, _, _) = self.ensure_prefill_workspace(shifted_ids.len())?;
            let _ = self.run_mtp_prefill_range(mtp_start_pos, &shifted_ids, p_nrm, true)?;

            // Deferred Single-Pass History Snapshot (0.0 ms on TTFT! 0 extra 8 GB weight passes!)
            // Now MTP KV cache is populated through prompt_len-1 AND rb.rec_final/conv_snaps hold exact hist state!
            if pf.deferred_hist_snap > 0 {
                let snap = self.capture_pinned_snapshot(pf.deferred_hist_snap, self.s_live_last_h, true)?;
                self.insert_prefix_snapshot(pids[..pf.deferred_hist_snap].to_vec(), snap);
            }

            // Immediately enqueue d2..dK + first Verify M=4 CUDA Graph (ZERO intermediate sync!)
            let mut cur_tok = first_tok;
            let mut cur_pos = prompt_len;
            let mut emitted = 1usize;
            self.enqueue_mtp_draft_and_verify(cur_tok, cur_pos)?;

            while emitted < max_new_tokens {
                // Single sync per cycle: reads BOTH drafts d1..dK and verified v_toks[0..M]!
                self.cuda.sync()?;
                let (draft_k, _verify_m, drafts, v_toks) = {
                    let m = self.mtp.as_ref().unwrap();
                    let mut d = [0i64; 3];
                    let mut v = [0i64; 4];
                    unsafe {
                        for i in 0..m.draft_k {
                            d[i] = *m.h_draft_toks.add(i);
                        }
                        for i in 0..m.verify_m {
                            v[i] = *m.h_verify_toks.add(i);
                        }
                    }
                    (m.draft_k, m.verify_m, d, v)
                };

                // Exact Greedy Verification
                let mut accepted = 0usize;
                while accepted < draft_k && drafts[accepted] == v_toks[accepted] {
                    accepted += 1;
                }
                let num_new = accepted + 1;
                if let Some(m) = self.mtp.as_mut() {
                    m.total_drafted += draft_k as u64;
                    m.total_accepted += accepted as u64;
                    m.total_cycles += 1;
                }

                // Determine if we stop within this cycle
                let mut hit_stop = false;
                let mut stop_at = num_new;
                for i in 0..num_new {
                    if emitted + i + 1 > max_new_tokens {
                        stop_at = i;
                        hit_stop = true;
                        break;
                    }
                    if self.tok.is_eos(v_toks[i]) && !ignore_eos {
                        stop_at = i + 1;
                        hit_stop = true;
                        break;
                    }
                    if emitted + i + 1 == max_new_tokens {
                        stop_at = i + 1;
                        hit_stop = true;
                        break;
                    }
                }

                // Append committed input tokens to Tier 0 live_tokens tracker
                if stop_at > 0 {
                    self.live_tokens.push(cur_tok);
                    for i in 0..(stop_at - 1) {
                        self.live_tokens.push(v_toks[i]);
                    }
                }

                // Enqueue ENTIRE next GPU pipeline in ONLY 2 CUDA GRAPH LAUNCHES BEFORE streaming SSE!
                if !hit_stop {
                    self.enqueue_mtp_commit_and_next_verify(accepted, &v_toks, cur_pos)?;
                    cur_pos += num_new;
                    cur_tok = v_toks[accepted];
                }

                // Stream all accepted + bonus tokens on CPU while GPU runs the entire next cycle!
                let mut client_abort = false;
                for i in 0..stop_at {
                    emitted += 1;
                    let tid = v_toks[i];
                    let is_e = self.tok.is_eos(tid) && !ignore_eos;
                    let finish = if is_e {
                        Some("stop")
                    } else if emitted >= max_new_tokens {
                        Some("length")
                    } else {
                        None
                    };
                    let cont = on_token(StreamToken {
                        token_id: tid,
                        text: if is_e { String::new() } else { self.tok.decode_single(tid) },
                        prompt_tokens: prompt_len,
                        cached_tokens,
                        cache_tier,
                        cache_restore_ms,
                        completion_tokens: emitted,
                        finish_reason: finish,
                    });
                    if is_e || !cont {
                        client_abort = true;
                        break;
                    }
                }

                if hit_stop {
                    // Cleanly commit exact GPU state for Tier 0 Hot-Cache (`live_tokens`)!
                    if stop_at > 0 {
                        let commit_idx = stop_at - 1;
                        self.commit_deltanet_states(commit_idx)?;
                        let (mtp_tid_ptr, v_nrm_ptr) = {
                            let m = self.mtp.as_ref().unwrap();
                            (m.mtp_tok_ids, m.v_normed)
                        };
                        let last_h_src = v_nrm_ptr + (commit_idx * 2560 * 2) as u64;
                        self.cuda.dtod_async(self.s_live_last_h, last_h_src, 2560 * 2)?;
                        if commit_idx > 0 {
                            self.cuda.htod_async(mtp_tid_ptr, v_toks.as_ptr() as *const u8, commit_idx * 8)?;
                            self.run_mtp_step_small(commit_idx, cur_pos, mtp_tid_ptr, v_nrm_ptr)?;
                        }
                        self.live_has_mtp = true;
                    } else {
                        self.live_tokens.clear();
                        self.live_has_mtp = false;
                    }
                    self.cuda.sync()?;
                    break;
                }
                if client_abort {
                    // Next cycle was already launched asynchronously; invalidate live_tokens safely
                    self.live_tokens.clear();
                    self.live_has_mtp = false;
                    self.cuda.sync()?;
                    break;
                }
            }
            return Ok(());
        }

        // Standard M=1 CUDA Graph Decode Path (when MTP K=0)
        // Prime GPU scalars for CUDA Graph decode + launch token #2 BEFORE yielding token #1!
        let pos_i32 = prompt_len as i32;
        self.cuda.htod_async(self.s_pos, &pos_i32 as *const i32 as *const u8, 4)?;
        self.cuda.htod_async(self.s_token_id, &first_tok as *const i64 as *const u8, 8)?;
        self.cuda.launch_graph(self.decode_graph)?;
        self.cuda.dtoh_async(self.h_tok_slots[0] as *mut u8, self.s_next_token_id, 8)?;
        self.cuda.record_event(self.ev_slots[0])?;

        // Yield first token while GPU is ALREADY executing token #2!
        let keep_going = on_token(StreamToken {
            token_id: first_tok,
            text: self.tok.decode_single(first_tok),
            prompt_tokens: prompt_len,
            cached_tokens,
            cache_tier,
            cache_restore_ms,
            completion_tokens: 1,
            finish_reason: None,
        });
        if !keep_going {
            self.live_tokens.clear();
            self.cuda.sync()?;
            return Ok(());
        }

        for step in 1..max_new_tokens {
            let cur = (step - 1) & 1;
            let nxt = step & 1;
            let has_next = step + 1 < max_new_tokens;

            if has_next {
                self.cuda.launch_graph(self.decode_graph)?;
                self.cuda.dtoh_async(self.h_tok_slots[nxt] as *mut u8, self.s_next_token_id, 8)?;
                self.cuda.record_event(self.ev_slots[nxt])?;
            }

            self.cuda.sync_event(self.ev_slots[cur])?;
            let tid = unsafe { *self.h_tok_slots[cur] };
            let is_eos = self.tok.is_eos(tid) && !ignore_eos;

            let finish = if is_eos {
                Some("stop")
            } else if step == max_new_tokens - 1 {
                Some("length")
            } else {
                None
            };

            let cont = on_token(StreamToken {
                token_id: tid,
                text: if is_eos { String::new() } else { self.tok.decode_single(tid) },
                prompt_tokens: prompt_len,
                cached_tokens,
                cache_tier,
                cache_restore_ms,
                completion_tokens: step + 1,
                finish_reason: finish,
            });
            if is_eos || !cont {
                self.live_tokens.clear();
                self.cuda.sync()?;
                break;
            }
        }
        // Standard M=1 speculative overlap may advance 1 extra step at end; clear Tier 0 safely (Tier 1 Pinned RAM still active!)
        self.live_tokens.clear();
        Ok(())
    }

    pub fn decode_64_profiled(&mut self, first_tok: i64, start_pos: usize, step_ms: &mut Vec<f64>) -> Result<()> {
        self.cuda.bind_context()?;
        let pos_i32 = start_pos as i32;
        self.cuda.htod_sync(self.s_pos, &pos_i32 as *const i32 as *const u8, 4)?;
        self.cuda.htod_sync(self.s_token_id, &first_tok as *const i64 as *const u8, 8)?;
        self.cuda.sync()?;
        for _ in 0..64 {
            let ts = Instant::now();
            self.cuda.launch_graph(self.decode_graph)?;
            self.cuda.dtoh_async(self.h_tok_slots[0] as *mut u8, self.s_next_token_id, 8)?;
            self.cuda.sync()?;
            step_ms.push(ts.elapsed().as_secs_f64() * 1000.0);
        }
        Ok(())
    }

    pub fn decode_64_mtp_profiled(&mut self, prompt_ids: &[i64], first_tok: i64) -> Result<(usize, f64)> {
        self.cuda.bind_context()?;
        let t0 = Instant::now();
        let prompt_len = prompt_ids.len();

        let mut shifted = Vec::with_capacity(prompt_len);
        shifted.extend_from_slice(&prompt_ids[1..]);
        shifted.push(first_tok);
        let (_, _, p_nrm, _, _, _, _, _) = self.ensure_prefill_workspace(shifted.len())?;
        let _ = self.run_mtp_prefill_range(0, &shifted, p_nrm, true)?;

        let mut cur_tok = first_tok;
        let mut cur_pos = prompt_len;
        let mut emitted = 0usize;
        let mut cycles = 0usize;

        self.enqueue_mtp_draft_and_verify(cur_tok, cur_pos)?;

        while emitted < 64 {
            self.cuda.sync()?;
            let (draft_k, drafts, v_toks) = {
                let m = self.mtp.as_ref().unwrap();
                let mut d = [0i64; 3];
                let mut v = [0i64; 4];
                unsafe {
                    for i in 0..m.draft_k {
                        d[i] = *m.h_draft_toks.add(i);
                    }
                    for i in 0..m.verify_m {
                        v[i] = *m.h_verify_toks.add(i);
                    }
                }
                (m.draft_k, d, v)
            };
            let mut accepted = 0usize;
            while accepted < draft_k && drafts[accepted] == v_toks[accepted] {
                accepted += 1;
            }
            let num_new = accepted + 1;
            if let Some(m) = self.mtp.as_mut() {
                m.total_drafted += draft_k as u64;
                m.total_accepted += accepted as u64;
                m.total_cycles += 1;
            }
            emitted += num_new;
            cycles += 1;
            if emitted >= 64 {
                break;
            }

            self.enqueue_mtp_commit_and_next_verify(accepted, &v_toks, cur_pos)?;
            cur_pos += num_new;
            cur_tok = v_toks[accepted];
        }
        self.cuda.sync()?;
        Ok((cycles, t0.elapsed().as_secs_f64() * 1000.0))
    }
}

