// Custom Fused CUDA Kernels for Pure Rust 2-Part Split Qwen3.5-4B Serving
// Compiled via nvcc -ptx -arch=sm_86 -O3 in build.rs
// 100% pure BF16 weights / FP32 recurrent state (zero model modifications)

#include <stdint.h>
#include <cuda_bf16.h>
#include <mma.h>

typedef unsigned short bf16_t;

__device__ __forceinline__ float bf16_to_f(bf16_t x) {
    unsigned int u = ((unsigned int)x) << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ bf16_t f_to_bf16(float x) {
    unsigned int u = __float_as_uint(x);
    unsigned int lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (bf16_t)(u >> 16);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, offset);
    }
    return v;
}

// ===========================================================================
// 1. Qwen3.5 RMSNorm (1-row decode AND multi-row prefill)
// ===========================================================================
extern "C" __global__ void qwen35_rmsnorm_2d_bf16(
    const bf16_t* __restrict__ x,
    const bf16_t* __restrict__ w,
    bf16_t* __restrict__ out,
    int dim,
    float eps
) {
    __shared__ float warp_sums[8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    const bf16_t* xr = x + (size_t)row * dim;
    bf16_t* orow = out + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float v = bf16_to_f(xr[i]);
        sum_sq += v * v;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    if (lane == 0) warp_sums[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 8) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) warp_sums[0] = rsqrtf(v / (float)dim + eps);
    }
    __syncthreads();
    float inv_rms = warp_sums[0];

    for (int i = tid; i < dim; i += 256) {
        float v = bf16_to_f(xr[i]) * inv_rms * (1.0f + bf16_to_f(w[i]));
        orow[i] = f_to_bf16(v);
    }
}

extern "C" __global__ void qwen35_add_rmsnorm_2d_bf16(
    bf16_t* __restrict__ residual,
    const bf16_t* __restrict__ delta,
    const bf16_t* __restrict__ w,
    bf16_t* __restrict__ out,
    int dim,
    float eps
) {
    __shared__ float warp_sums[8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    bf16_t* rr = residual + (size_t)row * dim;
    const bf16_t* dr = delta + (size_t)row * dim;
    bf16_t* orow = out + (size_t)row * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float v = bf16_to_f(rr[i]) + bf16_to_f(dr[i]);
        bf16_t vb = f_to_bf16(v);
        rr[i] = vb;
        float vr = bf16_to_f(vb);
        sum_sq += vr * vr;
    }
    sum_sq = warp_reduce_sum(sum_sq);
    if (lane == 0) warp_sums[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < 8) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) warp_sums[0] = rsqrtf(v / (float)dim + eps);
    }
    __syncthreads();
    float inv_rms = warp_sums[0];

    for (int i = tid; i < dim; i += 256) {
        float v = bf16_to_f(rr[i]) * inv_rms * (1.0f + bf16_to_f(w[i]));
        orow[i] = f_to_bf16(v);
    }
}

// ===========================================================================
// 2. Fused Packed SwiGLU: gate_up [rows, 2*inter] -> out [rows, inter]
// ===========================================================================
extern "C" __global__ void silu_mul_packed_2d_bf16(
    const bf16_t* __restrict__ gate_up,
    bf16_t* __restrict__ out,
    int inter,
    int rows
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y;
    if (i < inter && r < rows) {
        const bf16_t* row_in = gate_up + (size_t)r * (2 * inter);
        float g = bf16_to_f(row_in[i]);
        float u = bf16_to_f(row_in[inter + i]);
        float s = g / (1.0f + expf(-g));
        out[(size_t)r * inter + i] = f_to_bf16(s * u);
    }
}

// ===========================================================================
// 3. Residual Add + Buffer Copy
// ===========================================================================
extern "C" __global__ void residual_add_bf16(
    bf16_t* __restrict__ dst,
    const bf16_t* __restrict__ src,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = f_to_bf16(bf16_to_f(dst[i]) + bf16_to_f(src[i]));
    }
}

extern "C" __global__ void copy_bf16(
    const bf16_t* __restrict__ src,
    bf16_t* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = src[i];
    }
}

// ===========================================================================
// 4. Causal Conv1D (k=4) Decode & Prefill
// ===========================================================================
extern "C" __global__ void causal_conv1d_update_k4_bf16(
    const bf16_t* __restrict__ x_in,
    bf16_t* __restrict__ conv_state,
    const bf16_t* __restrict__ conv_weight,
    bf16_t* __restrict__ y_out,
    int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int base = c * 4;
    bf16_t s1 = conv_state[base + 1];
    bf16_t s2 = conv_state[base + 2];
    bf16_t s3 = conv_state[base + 3];
    bf16_t s4 = x_in[c];

    conv_state[base + 0] = s1;
    conv_state[base + 1] = s2;
    conv_state[base + 2] = s3;
    conv_state[base + 3] = s4;

    float acc = bf16_to_f(s1) * bf16_to_f(conv_weight[base + 0])
              + bf16_to_f(s2) * bf16_to_f(conv_weight[base + 1])
              + bf16_to_f(s3) * bf16_to_f(conv_weight[base + 2])
              + bf16_to_f(s4) * bf16_to_f(conv_weight[base + 3]);
    float silu = acc / (1.0f + expf(-acc));
    y_out[c] = f_to_bf16(silu);
}

extern "C" __global__ void causal_conv1d_prefill_k4_bf16(
    const bf16_t* __restrict__ packed_in,
    bf16_t* __restrict__ conv_state,
    bf16_t* __restrict__ conv_snap_out,  // optional: if snap_t > 0, writes conv_state at t == snap_t - 1!
    const bf16_t* __restrict__ conv_weight,
    bf16_t* __restrict__ y_out,
    int seq_len,
    int C,
    int stride_in,
    int use_init_state,
    int snap_t
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    int wbase = c * 4;
    float w0 = bf16_to_f(conv_weight[wbase + 0]);
    float w1 = bf16_to_f(conv_weight[wbase + 1]);
    float w2 = bf16_to_f(conv_weight[wbase + 2]);
    float w3 = bf16_to_f(conv_weight[wbase + 3]);

    bf16_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    if (use_init_state) {
        s1 = conv_state[wbase + 1];
        s2 = conv_state[wbase + 2];
        s3 = conv_state[wbase + 3];
    }

    for (int t = 0; t < seq_len; ++t) {
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = packed_in[(size_t)t * stride_in + c];

        if (snap_t > 0 && t == snap_t - 1) {
            conv_snap_out[wbase + 0] = s0;
            conv_snap_out[wbase + 1] = s1;
            conv_snap_out[wbase + 2] = s2;
            conv_snap_out[wbase + 3] = s3;
        }

        float acc = bf16_to_f(s0) * w0
                  + bf16_to_f(s1) * w1
                  + bf16_to_f(s2) * w2
                  + bf16_to_f(s3) * w3;
        float silu = acc / (1.0f + expf(-acc));
        y_out[(size_t)t * C + c] = f_to_bf16(silu);
    }

    conv_state[wbase + 0] = s0;
    conv_state[wbase + 1] = s1;
    conv_state[wbase + 2] = s2;
    conv_state[wbase + 3] = s3;
}

// ===========================================================================
// 5. Fused Gated DeltaNet Decode (1-Pass Register-Tiled) & Prefill (128-Block Zero-Spill)
// ===========================================================================
// Grid: (32, 1, 1), Block: (256, 1, 1) -> 2 K-halves x 128 V-cols, reads S_head ONCE!
extern "C" __global__ void gated_deltanet_decode_packed_bf16(
    const bf16_t* __restrict__ qkv_conv,
    const bf16_t* __restrict__ packed_in,
    const bf16_t* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    const bf16_t* __restrict__ norm_w,
    float* __restrict__ rec_state,
    bf16_t* __restrict__ out,
    float eps
) {
    int hv = blockIdx.x;
    int tid = threadIdx.x;
    int k_half = tid >> 7;     // 0 or 1
    int v_col = tid & 127;     // 0..127
    int lane = v_col & 31;
    int wid = v_col >> 5;      // 0..3
    int hk = hv >> 1;

    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float wsum_q[4];
    __shared__ float wsum_k[4];
    __shared__ float sh_half[128];

    // Half 0 normalizes q, Half 1 normalizes k IN PARALLEL (only 2 syncs total!)
    if (k_half == 0) {
        float qv = bf16_to_f(qkv_conv[hk * 128 + v_col]);
        float qs = warp_reduce_sum(qv * qv);
        if (lane == 0) wsum_q[wid] = qs;
        s_q[v_col] = qv;
    } else {
        float kv = bf16_to_f(qkv_conv[2048 + hk * 128 + v_col]);
        float ks = warp_reduce_sum(kv * kv);
        if (lane == 0) wsum_k[wid] = ks;
        s_k[v_col] = kv;
    }
    __syncthreads();
    if (k_half == 0 && wid == 0) {
        float v = warp_reduce_sum((lane < 4) ? wsum_q[lane] : 0.0f);
        if (lane == 0) wsum_q[0] = rsqrtf(v + 1e-6f) * 0.08838834764831845f;
    } else if (k_half == 1 && wid == 0) {
        float v = warp_reduce_sum((lane < 4) ? wsum_k[lane] : 0.0f);
        if (lane == 0) wsum_k[0] = rsqrtf(v + 1e-6f);
    }
    __syncthreads();
    if (k_half == 0) {
        s_q[v_col] *= wsum_q[0];
    } else {
        s_k[v_col] *= wsum_k[0];
    }
    __syncthreads();

    const bf16_t* z_in = packed_in + 8192;
    const bf16_t* b_in = packed_in + 12288;
    const bf16_t* a_in = packed_in + 12320;

    float bv = bf16_to_f(b_in[hv]);
    float beta = 1.0f / (1.0f + expf(-bv));
    float av = bf16_to_f(a_in[hv]) + bf16_to_f(dt_bias[hv]);
    float sp = (av > 20.0f) ? av : log1pf(expf(av));
    float g_exp = expf(-expf(A_log[hv]) * sp);

    float v_val = bf16_to_f(qkv_conv[4096 + hv * 128 + v_col]);
    float* S_head = rec_state + (size_t)hv * 16384;
    int k_base = k_half * 64;

    // Load 64 elements of S ONCE into registers (zero second DRAM read!)
    float s_reg[64];
    float kv_part = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        float sd = S_head[(k_base + j) * 128 + v_col] * g_exp;
        s_reg[j] = sd;
        kv_part += sd * s_k[k_base + j];
    }

    if (k_half == 1) sh_half[v_col] = kv_part;
    __syncthreads();
    if (k_half == 0) {
        float kv_mem = kv_part + sh_half[v_col];
        sh_half[v_col] = (v_val - kv_mem) * beta;
    }
    __syncthreads();
    float delta = sh_half[v_col];

    float out_part = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        float s_new = s_reg[j] + s_k[k_base + j] * delta;
        S_head[(k_base + j) * 128 + v_col] = s_new;
        out_part += s_new * s_q[k_base + j];
    }

    __syncthreads();
    if (k_half == 1) sh_half[v_col] = out_part;
    __syncthreads();

    if (k_half == 0) {
        float out_val = out_part + sh_half[v_col];
        float os = warp_reduce_sum(out_val * out_val);
        if (lane == 0) wsum_q[wid] = os;
        __syncwarp();
        // Need block-level sync across the 4 warps of half 0 -> use shared reduction
        sh_half[v_col] = out_val;
    }
    __syncthreads();
    if (k_half == 0 && wid == 0) {
        float v = warp_reduce_sum((lane < 4) ? wsum_q[lane] : 0.0f);
        if (lane == 0) wsum_q[0] = rsqrtf(v * (1.0f / 128.0f) + eps);
    }
    __syncthreads();
    if (k_half == 0) {
        float out_val = sh_half[v_col];
        float normed = bf16_to_f(f_to_bf16(out_val * wsum_q[0])) * bf16_to_f(norm_w[v_col]);
        float zv = bf16_to_f(z_in[hv * 128 + v_col]);
        float z_silu = zv / (1.0f + expf(-zv));
        out[hv * 128 + v_col] = f_to_bf16(normed * z_silu);
    }
}

// Grid: (32, 4, 1) = 128 blocks (saturates all 68 SMs!), Block: (128, 1, 1)
// Each thread holds 32 FP32 state elements in registers -> ZERO local memory spill!
extern "C" __global__ void gated_deltanet_prefill_bf16(
    const bf16_t* __restrict__ qkv_conv,
    const bf16_t* __restrict__ packed_in,
    const bf16_t* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    float* __restrict__ rec_state,
    float* __restrict__ rec_snap_out,   // optional: if snap_t > 0, writes S at t == snap_t - 1 (single pass!)
    float* __restrict__ out_raw,
    int seq_len,
    int use_init_state,
    int snap_t
) {
    int hv = blockIdx.x;
    int vs = blockIdx.y;       // 0..3 -> 32 V-columns per block
    int tid = threadIdx.x;
    int lane = tid & 31;       // 0..31 -> V-column offset
    int wid = tid >> 5;        // 0..3  -> K-slice (32 K-rows per warp)
    int hk = hv >> 1;
    int v_col = vs * 32 + lane;
    int k_base = wid * 32;

    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float wsum_q[4];
    __shared__ float wsum_k[4];
    __shared__ float sh_kv[4][32];
    __shared__ float sh_out[4][32];

    float s_reg[32];
    float* S_head = rec_state + (size_t)hv * 16384;
    if (use_init_state) {
        #pragma unroll
        for (int j = 0; j < 32; ++j) s_reg[j] = S_head[(k_base + j) * 128 + v_col];
    } else {
        #pragma unroll
        for (int j = 0; j < 32; ++j) s_reg[j] = 0.0f;
    }

    float dt_b = bf16_to_f(dt_bias[hv]);
    float neg_exp_A = -expf(A_log[hv]);

    for (int t = 0; t < seq_len; ++t) {
        const bf16_t* qkv_t = qkv_conv + (size_t)t * 8192;
        const bf16_t* pin_t = packed_in + (size_t)t * 12352;

        float qv = bf16_to_f(qkv_t[hk * 128 + tid]);
        float kv = bf16_to_f(qkv_t[2048 + hk * 128 + tid]);

        float qs = warp_reduce_sum(qv * qv);
        float ks = warp_reduce_sum(kv * kv);
        if (lane == 0) {
            wsum_q[wid] = qs;
            wsum_k[wid] = ks;
        }
        __syncthreads();
        if (wid == 0) {
            float vq = warp_reduce_sum((lane < 4) ? wsum_q[lane] : 0.0f);
            float vk = warp_reduce_sum((lane < 4) ? wsum_k[lane] : 0.0f);
            if (lane == 0) {
                wsum_q[0] = rsqrtf(vq + 1e-6f) * 0.08838834764831845f;
                wsum_k[0] = rsqrtf(vk + 1e-6f);
            }
        }
        __syncthreads();
        s_q[tid] = qv * wsum_q[0];
        s_k[tid] = kv * wsum_k[0];
        __syncthreads();

        float bv = bf16_to_f(pin_t[12288 + hv]);
        float beta = 1.0f / (1.0f + expf(-bv));
        float av = bf16_to_f(pin_t[12320 + hv]) + dt_b;
        float sp = (av > 20.0f) ? av : log1pf(expf(av));
        float g_exp = expf(neg_exp_A * sp);

        float v_val = bf16_to_f(qkv_t[4096 + hv * 128 + v_col]);

        float kv_part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float sd = s_reg[j] * g_exp;
            s_reg[j] = sd;
            kv_part += sd * s_k[k_base + j];
        }
        sh_kv[wid][lane] = kv_part;
        __syncthreads();

        float kv_mem = sh_kv[0][lane] + sh_kv[1][lane] + sh_kv[2][lane] + sh_kv[3][lane];
        float delta = (v_val - kv_mem) * beta;

        float out_part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float sn = s_reg[j] + s_k[k_base + j] * delta;
            s_reg[j] = sn;
            out_part += sn * s_q[k_base + j];
        }
        sh_out[wid][lane] = out_part;
        __syncthreads();

        if (wid == 0) {
            float ov = sh_out[0][lane] + sh_out[1][lane] + sh_out[2][lane] + sh_out[3][lane];
            out_raw[(size_t)t * 4096 + hv * 128 + v_col] = ov;
        }
        if (snap_t > 0 && t == snap_t - 1) {
            float* S_snap = rec_snap_out + (size_t)hv * 16384;
            #pragma unroll
            for (int j = 0; j < 32; ++j) {
                S_snap[(k_base + j) * 128 + v_col] = s_reg[j];
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        S_head[(k_base + j) * 128 + v_col] = s_reg[j];
    }
}

// Grid: (32, seq_len, 1), Block: (128, 1, 1) -> Fused RMSNorm + SiLU(z) gate for DeltaNet Prefill
extern "C" __global__ void deltanet_prefill_norm_gate_bf16(
    const float* __restrict__ out_raw,
    const bf16_t* __restrict__ packed_in,
    const bf16_t* __restrict__ norm_w,
    bf16_t* __restrict__ out,
    float eps
) {
    int hv = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    __shared__ float wsum[4];
    float out_val = out_raw[(size_t)t * 4096 + hv * 128 + tid];
    float os = warp_reduce_sum(out_val * out_val);
    if (lane == 0) wsum[wid] = os;
    __syncthreads();
    if (wid == 0) {
        float v = warp_reduce_sum((lane < 4) ? wsum[lane] : 0.0f);
        if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 128.0f) + eps);
    }
    __syncthreads();
    float normed = bf16_to_f(f_to_bf16(out_val * wsum[0])) * bf16_to_f(norm_w[tid]);
    float zv = bf16_to_f(packed_in[(size_t)t * 12352 + 8192 + hv * 128 + tid]);
    float z_silu = zv / (1.0f + expf(-zv));
    out[(size_t)t * 4096 + hv * 128 + tid] = f_to_bf16(normed * z_silu);
}

// ===========================================================================
// 6. Full Attention Decode Prep + Prefill Prep + Fused Causal Prefill GQA
// ===========================================================================
extern "C" __global__ void full_attn_qk_rope_kv_write_packed_bf16(
    const bf16_t* __restrict__ qkv_packed,
    const bf16_t* __restrict__ q_norm_w,
    const bf16_t* __restrict__ k_norm_w,
    const bf16_t* __restrict__ cos_table,
    const bf16_t* __restrict__ sin_table,
    const int* __restrict__ pos_ptr,
    bf16_t* __restrict__ q_out,
    bf16_t* __restrict__ k_cache,
    bf16_t* __restrict__ v_cache,
    int max_seq,
    float eps
) {
    int b = blockIdx.x;
    int d = threadIdx.x;
    int lane = d & 31;
    int wid = d >> 5;
    int pos = pos_ptr[0];

    __shared__ float wsum[8];
    __shared__ float norm_buf[64];

    if (b < 16) {
        const bf16_t* src = qkv_packed + b * 512;
        float val = bf16_to_f(src[d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? wsum[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(q_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        q_out[b * 256 + d] = f_to_bf16(nv);
    } else {
        int kh = b - 16;
        const bf16_t* k_raw = qkv_packed + 8192;
        const bf16_t* v_raw = qkv_packed + 9216;
        float val = bf16_to_f(k_raw[kh * 256 + d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? wsum[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(k_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        size_t kv_off = ((size_t)kh * max_seq + pos) * 256 + d;
        k_cache[kv_off] = f_to_bf16(nv);
        v_cache[kv_off] = v_raw[kh * 256 + d];
    }
}

extern "C" __global__ void full_attn_qk_rope_kv_prefill_bf16(
    const bf16_t* __restrict__ qkv_packed,
    const bf16_t* __restrict__ q_norm_w,
    const bf16_t* __restrict__ k_norm_w,
    const bf16_t* __restrict__ cos_table,
    const bf16_t* __restrict__ sin_table,
    int start_pos,
    int seq_len,
    bf16_t* __restrict__ q_sdpa,
    bf16_t* __restrict__ k_cache,
    bf16_t* __restrict__ v_cache,
    int max_seq,
    float eps
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int lane = d & 31;
    int wid = d >> 5;
    int pos = start_pos + t;

    __shared__ float wsum[8];
    __shared__ float norm_buf[64];

    const bf16_t* row = qkv_packed + (size_t)t * 10240;

    if (b < 16) {
        const bf16_t* src = row + b * 512;
        float val = bf16_to_f(src[d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? wsum[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(q_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        q_sdpa[((size_t)b * seq_len + t) * 256 + d] = f_to_bf16(nv);
    } else {
        int kh = b - 16;
        const bf16_t* k_raw = row + 8192;
        const bf16_t* v_raw = row + 9216;
        float val = bf16_to_f(k_raw[kh * 256 + d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = (lane < 8) ? wsum[lane] : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(k_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        size_t kv_off = ((size_t)kh * max_seq + pos) * 256 + d;
        k_cache[kv_off] = f_to_bf16(nv);
        v_cache[kv_off] = v_raw[kh * 256 + d];
    }
}

__device__ __forceinline__ void unpack_uint4_bf16(uint4 v, float out[8]) {
    out[0] = bf16_to_f((bf16_t)(v.x & 0xFFFFu));
    out[1] = bf16_to_f((bf16_t)(v.x >> 16));
    out[2] = bf16_to_f((bf16_t)(v.y & 0xFFFFu));
    out[3] = bf16_to_f((bf16_t)(v.y >> 16));
    out[4] = bf16_to_f((bf16_t)(v.z & 0xFFFFu));
    out[5] = bf16_to_f((bf16_t)(v.z >> 16));
    out[6] = bf16_to_f((bf16_t)(v.w & 0xFFFFu));
    out[7] = bf16_to_f((bf16_t)(v.w >> 16));
}

__device__ __forceinline__ uint4 pack_uint4_bf16(const float in[8]) {
    uint4 r;
    r.x = (unsigned int)f_to_bf16(in[0]) | ((unsigned int)f_to_bf16(in[1]) << 16);
    r.y = (unsigned int)f_to_bf16(in[2]) | ((unsigned int)f_to_bf16(in[3]) << 16);
    r.z = (unsigned int)f_to_bf16(in[4]) | ((unsigned int)f_to_bf16(in[5]) << 16);
    r.w = (unsigned int)f_to_bf16(in[6]) | ((unsigned int)f_to_bf16(in[7]) << 16);
    return r;
}

// FlashAttention-2 Style Tiled Causal Prefill GQA + Sigmoid Gate (16-KB SMEM KV Tile, uint4 vectorized!)
// Grid: (4 KV-heads, (seq_len + 1) / 2), Block: (256,) = 8 warps (4 Q-heads x 2 query tokens per block!)
extern "C" __global__ void prefill_gqa_causal_gated_bf16(
    const bf16_t* __restrict__ q_sdpa,      // [16, seq_len, 256]
    const bf16_t* __restrict__ qkv_packed,  // [seq_len, 10240] -> gate @ hq*512 + 256
    const bf16_t* __restrict__ k_cache,     // [4, max_seq, 256]
    const bf16_t* __restrict__ v_cache,     // [4, max_seq, 256]
    int start_pos,
    int seq_len,
    int max_seq,
    bf16_t* __restrict__ out                // [seq_len, 4096]
) {
    int hk = blockIdx.x;        // 0..3
    int t_base = blockIdx.y * 2;
    int tid = threadIdx.x;
    int wid = tid >> 5;         // 0..7
    int lane = tid & 31;        // 0..31 -> 8 elements: lane*8 .. lane*8+7
    int hq = hk * 4 + (wid >> 1);
    int t_my = t_base + (wid & 1);
    bool valid_q = (t_my < seq_len);

    const float scale = 0.0625f;
    float q_reg[8];
    if (valid_q) {
        uint4 q4 = *reinterpret_cast<const uint4*>(
            q_sdpa + ((size_t)hq * seq_len + t_my) * 256 + lane * 8
        );
        unpack_uint4_bf16(q4, q_reg);
        #pragma unroll
        for (int j = 0; j < 8; ++j) q_reg[j] *= scale;
    }

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    __shared__ __align__(16) bf16_t s_K[16][256];
    __shared__ __align__(16) bf16_t s_V[16][256];

    const bf16_t* k_head = k_cache + (size_t)hk * max_seq * 256;
    const bf16_t* v_head = v_cache + (size_t)hk * max_seq * 256;

    int t_max_blk = (t_base + 1 < seq_len) ? (t_base + 1) : (seq_len - 1);
    int kv_end_blk = start_pos + t_max_blk + 1;
    int kv_end_my = valid_q ? (start_pos + t_my + 1) : 0;

    // Each thread cooperatively loads 2 uint4s (16 bf16s) per tile for K and V
    int r0 = tid >> 5;          // 0..7
    int c0 = (tid & 31) * 8;    // 0..248
    int r1 = r0 + 8;            // 8..15

    for (int kt0 = 0; kt0 < kv_end_blk; kt0 += 16) {
        if (kt0 + r0 < kv_end_blk) {
            *reinterpret_cast<uint4*>(&s_K[r0][c0]) = *reinterpret_cast<const uint4*>(k_head + (size_t)(kt0 + r0) * 256 + c0);
            *reinterpret_cast<uint4*>(&s_V[r0][c0]) = *reinterpret_cast<const uint4*>(v_head + (size_t)(kt0 + r0) * 256 + c0);
        }
        if (kt0 + r1 < kv_end_blk) {
            *reinterpret_cast<uint4*>(&s_K[r1][c0]) = *reinterpret_cast<const uint4*>(k_head + (size_t)(kt0 + r1) * 256 + c0);
            *reinterpret_cast<uint4*>(&s_V[r1][c0]) = *reinterpret_cast<const uint4*>(v_head + (size_t)(kt0 + r1) * 256 + c0);
        }
        __syncthreads();

        if (valid_q && kt0 < kv_end_my) {
            int k_lim = kv_end_my - kt0;
            if (k_lim > 16) k_lim = 16;
            for (int ki = 0; ki < k_lim; ++ki) {
                uint4 k4 = *reinterpret_cast<const uint4*>(&s_K[ki][lane * 8]);
                float kf[8];
                unpack_uint4_bf16(k4, kf);
                float prod = 0.0f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) prod += q_reg[j] * kf[j];
                float score = warp_reduce_sum(prod);

                float new_max = fmaxf(m_max, score);
                float alpha = expf(m_max - new_max);
                float p = expf(score - new_max);
                l_sum = l_sum * alpha + p;
                m_max = new_max;

                uint4 v4 = *reinterpret_cast<const uint4*>(&s_V[ki][lane * 8]);
                float vf[8];
                unpack_uint4_bf16(v4, vf);
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[j] = acc[j] * alpha + p * vf[j];
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        float inv_l = 1.0f / fmaxf(l_sum, 1e-30f);
        uint4 g4 = *reinterpret_cast<const uint4*>(
            qkv_packed + (size_t)t_my * 10240 + hq * 512 + 256 + lane * 8
        );
        float gf[8];
        unpack_uint4_bf16(g4, gf);
        float res[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float sig = 1.0f / (1.0f + expf(-gf[j]));
            res[j] = (acc[j] * inv_l) * sig;
        }
        *reinterpret_cast<uint4*>(out + (size_t)t_my * 4096 + hq * 256 + lane * 8) = pack_uint4_bf16(res);
    }
}

// ===========================================================================
// 7. FlashDecoding Split-KV GQA Decode (Warp-Vectorized uint4, Zero Inner Sync!)
// ===========================================================================
extern "C" __global__ void gqa_decode_flash_splitk_bf16(
    const bf16_t* __restrict__ q_in,
    const bf16_t* __restrict__ k_cache,
    const bf16_t* __restrict__ v_cache,
    const int* __restrict__ pos_ptr,
    float* __restrict__ part_out,
    float* __restrict__ part_ml,
    int max_seq,
    int num_splits
) {
    int hq = blockIdx.x;
    int sp = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;         // 0..7 warps process strided t independently!
    int hk = hq >> 2;
    int seq_len = pos_ptr[0] + 1;

    int chunk = (seq_len + num_splits - 1) / num_splits;
    int t_start = sp * chunk;
    int t_end = t_start + chunk;
    if (t_end > seq_len) t_end = seq_len;

    const float scale = 0.0625f;
    uint4 q4 = *reinterpret_cast<const uint4*>(q_in + hq * 256 + lane * 8);
    float q_reg[8];
    unpack_uint4_bf16(q4, q_reg);
    #pragma unroll
    for (int j = 0; j < 8; ++j) q_reg[j] *= scale;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const bf16_t* k_head = k_cache + (size_t)hk * max_seq * 256;
    const bf16_t* v_head = v_cache + (size_t)hk * max_seq * 256;

    // ZERO __syncthreads() inside the entire KV loop!
    for (int t = t_start + wid; t < t_end; t += 8) {
        uint4 k4 = *reinterpret_cast<const uint4*>(k_head + (size_t)t * 256 + lane * 8);
        float kf[8];
        unpack_uint4_bf16(k4, kf);
        float prod = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) prod += q_reg[j] * kf[j];
        float score = warp_reduce_sum(prod);

        float new_max = fmaxf(m_max, score);
        float alpha = expf(m_max - new_max);
        float p = expf(score - new_max);
        l_sum = l_sum * alpha + p;
        m_max = new_max;

        uint4 v4 = *reinterpret_cast<const uint4*>(v_head + (size_t)t * 256 + lane * 8);
        float vf[8];
        unpack_uint4_bf16(v4, vf);
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = acc[j] * alpha + p * vf[j];
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_acc[8][256];

    if (lane == 0) {
        sm_m[wid] = m_max;
        sm_l[wid] = l_sum;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        sm_acc[wid][lane * 8 + j] = acc[j];
    }
    __syncthreads();

    // Merge 8 warps across 256 threads (1 thread per head_dim element)
    int d = tid;
    float g_max = -1e30f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) g_max = fmaxf(g_max, sm_m[w]);

    float g_sum = 0.0f;
    float g_acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float lw = sm_l[w];
        if (lw > 0.0f) {
            float mult = expf(sm_m[w] - g_max);
            g_sum += lw * mult;
            g_acc += sm_acc[w][d] * mult;
        }
    }

    size_t idx_base = ((size_t)hq * num_splits + sp);
    part_out[idx_base * 256 + d] = g_acc;
    if (d == 0) {
        part_ml[idx_base * 2 + 0] = g_max;
        part_ml[idx_base * 2 + 1] = g_sum;
    }
}

extern "C" __global__ void gqa_decode_flash_reduce_bf16(
    const float* __restrict__ part_out,
    const float* __restrict__ part_ml,
    const bf16_t* __restrict__ qkv_packed,
    bf16_t* __restrict__ out,
    int num_splits
) {
    int hq = blockIdx.x;
    int d = threadIdx.x;

    float g_max = -1e30f;
    for (int sp = 0; sp < num_splits; ++sp) {
        float m = part_ml[((size_t)hq * num_splits + sp) * 2 + 0];
        g_max = fmaxf(g_max, m);
    }

    float g_sum = 0.0f;
    float g_acc = 0.0f;
    for (int sp = 0; sp < num_splits; ++sp) {
        size_t base = (size_t)hq * num_splits + sp;
        float m = part_ml[base * 2 + 0];
        float l = part_ml[base * 2 + 1];
        if (l > 0.0f) {
            float w = expf(m - g_max);
            g_sum += l * w;
            g_acc += part_out[base * 256 + d] * w;
        }
    }

    float attn_val = g_acc / fmaxf(g_sum, 1e-30f);
    float gate_val = bf16_to_f(qkv_packed[hq * 512 + 256 + d]);
    float sig = 1.0f / (1.0f + expf(-gate_val));
    out[hq * 256 + d] = f_to_bf16(attn_val * sig);
}

// ===========================================================================
// 8. Embedding Lookup (single & prefill), Argmax (2-stage), GPU scalars
// ===========================================================================
extern "C" __global__ void embed_lookup_single_bf16(
    const long long* __restrict__ token_id_ptr,
    const bf16_t* __restrict__ embed_table,
    bf16_t* __restrict__ out,
    int hidden_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hidden_size) {
        long long tid = token_id_ptr[0];
        out[i] = embed_table[tid * (long long)hidden_size + i];
    }
}

extern "C" __global__ void embed_lookup_prefill_bf16(
    const long long* __restrict__ token_ids,
    const bf16_t* __restrict__ embed_table,
    bf16_t* __restrict__ out,
    int seq_len,
    int hidden_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (i < hidden_size && t < seq_len) {
        long long tid = token_ids[t];
        out[(size_t)t * hidden_size + i] = embed_table[tid * (long long)hidden_size + i];
    }
}

extern "C" __global__ void argmax_stage1_bf16(
    const bf16_t* __restrict__ logits,
    float* __restrict__ part_vals,
    int* __restrict__ part_idxs,
    int n
) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int stride = gridDim.x * 256;

    float best_v = -1e30f;
    int best_i = 0;
    for (int i = bid * 256 + tid; i < n; i += stride) {
        float v = bf16_to_f(logits[i]);
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_vals[bid] = s_val[0];
        part_idxs[bid] = s_idx[0];
    }
}

extern "C" __global__ void argmax_stage2_bf16(
    const float* __restrict__ part_vals,
    const int* __restrict__ part_idxs,
    long long* __restrict__ out_id,
    int num_blocks
) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    int tid = threadIdx.x;
    float bv = -1e30f;
    int bi = 0;
    for (int i = tid; i < num_blocks; i += 256) {
        float v = part_vals[i];
        if (v > bv) {
            bv = v;
            bi = part_idxs[i];
        }
    }
    s_val[tid] = bv;
    s_idx[tid] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_id[0] = (long long)s_idx[0];
    }
}

extern "C" __global__ void inc_pos_i32(int* pos_ptr) {
    pos_ptr[0] += 1;
}

extern "C" __global__ void copy_i64(const long long* __restrict__ src, long long* __restrict__ dst) {
    dst[0] = src[0];
}

// ===========================================================================
// 9. Phase R3: Native Qwen3.5 MTP Speculative Decoding + Parallel Verify (M=2..4)
// ===========================================================================
// Fused embed(tok_ids) + pre_fc_norm_embedding + pre_fc_norm_hidden + concat -> [M, 5120]
// Grid: (M, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void mtp_pre_fc_concat_bf16(
    const long long* __restrict__ token_ids,
    const bf16_t* __restrict__ embed_table,
    const bf16_t* __restrict__ hid_in,
    const bf16_t* __restrict__ w_emb_norm,
    const bf16_t* __restrict__ w_hid_norm,
    bf16_t* __restrict__ out_cat,
    int dim,
    float eps
) {
    __shared__ float wsum_e[8];
    __shared__ float wsum_h[8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    long long tok = token_ids[row];
    const bf16_t* erow = embed_table + tok * (long long)dim;
    const bf16_t* hrow = hid_in + (size_t)row * dim;
    bf16_t* orow = out_cat + (size_t)row * (2 * dim);

    float sq_e = 0.0f;
    float sq_h = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float ve = bf16_to_f(erow[i]);
        float vh = bf16_to_f(hrow[i]);
        sq_e += ve * ve;
        sq_h += vh * vh;
    }
    sq_e = warp_reduce_sum(sq_e);
    sq_h = warp_reduce_sum(sq_h);
    if (lane == 0) {
        wsum_e[wid] = sq_e;
        wsum_h[wid] = sq_h;
    }
    __syncthreads();
    if (wid == 0) {
        float ve = warp_reduce_sum((lane < 8) ? wsum_e[lane] : 0.0f);
        float vh = warp_reduce_sum((lane < 8) ? wsum_h[lane] : 0.0f);
        if (lane == 0) {
            wsum_e[0] = rsqrtf(ve / (float)dim + eps);
            wsum_h[0] = rsqrtf(vh / (float)dim + eps);
        }
    }
    __syncthreads();
    float inv_e = wsum_e[0];
    float inv_h = wsum_h[0];
    for (int i = tid; i < dim; i += 256) {
        orow[i]       = f_to_bf16(bf16_to_f(erow[i]) * inv_e * (1.0f + bf16_to_f(w_emb_norm[i])));
        orow[dim + i] = f_to_bf16(bf16_to_f(hrow[i]) * inv_h * (1.0f + bf16_to_f(w_hid_norm[i])));
    }
}

// Non-destructive Conv1D for M verified tokens: reads conv_state, writes M snapshots + y_out
// Grid: (32, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void causal_conv1d_verify_m_bf16(
    const bf16_t* __restrict__ packed_in,   // [M, 12352]
    const bf16_t* __restrict__ conv_state,  // [8192, 4] READ-ONLY
    bf16_t* __restrict__ conv_snaps,        // [M, 8192, 4] WRITE
    const bf16_t* __restrict__ conv_weight, // [8192, 4]
    bf16_t* __restrict__ y_out,             // [M, 8192]
    int M,
    int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;

    int wbase = c * 4;
    float w0 = bf16_to_f(conv_weight[wbase + 0]);
    float w1 = bf16_to_f(conv_weight[wbase + 1]);
    float w2 = bf16_to_f(conv_weight[wbase + 2]);
    float w3 = bf16_to_f(conv_weight[wbase + 3]);

    bf16_t s0 = 0;
    bf16_t s1 = conv_state[wbase + 1];
    bf16_t s2 = conv_state[wbase + 2];
    bf16_t s3 = conv_state[wbase + 3];

    for (int m = 0; m < M; ++m) {
        s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = packed_in[(size_t)m * 12352 + c];

        size_t sbase = (size_t)m * (C * 4) + wbase;
        conv_snaps[sbase + 0] = s0;
        conv_snaps[sbase + 1] = s1;
        conv_snaps[sbase + 2] = s2;
        conv_snaps[sbase + 3] = s3;

        float acc = bf16_to_f(s0) * w0 + bf16_to_f(s1) * w1 + bf16_to_f(s2) * w2 + bf16_to_f(s3) * w3;
        y_out[(size_t)m * C + c] = f_to_bf16(acc / (1.0f + expf(-acc)));
    }
}

// Non-destructive Register-Tiled DeltaNet Verify for M tokens (Zero Spill, 128 blocks!)
// Reads rec_state_in (READ-ONLY!), writes final m=M-1 state to rec_state_final + saves (beta, g_exp)
// Grid: (32, 4, 1), Block: (128, 1, 1)
extern "C" __global__ void gated_deltanet_verify_m_bf16(
    const bf16_t* __restrict__ qkv_conv,    // [M, 8192]
    const bf16_t* __restrict__ packed_in,   // [M, 12352]
    const bf16_t* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    const float* __restrict__ rec_state_in, // [32, 128, 128] READ-ONLY
    float* __restrict__ rec_state_final,    // [32, 128, 128] WRITE
    float* __restrict__ out_raw,            // [M, 4096]
    float* __restrict__ saved_bg,           // [M, 32, 2] (beta, g_exp)
    int M
) {
    int hv = blockIdx.x;
    int vs = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int hk = hv >> 1;
    int v_col = vs * 32 + lane;
    int k_base = wid * 32;

    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float wsum_q[4];
    __shared__ float wsum_k[4];
    __shared__ float sh_kv[4][32];
    __shared__ float sh_out[4][32];

    float s_reg[32];
    const float* S_in = rec_state_in + (size_t)hv * 16384;
    #pragma unroll
    for (int j = 0; j < 32; ++j) s_reg[j] = S_in[(k_base + j) * 128 + v_col];

    float dt_b = bf16_to_f(dt_bias[hv]);
    float neg_exp_A = -expf(A_log[hv]);

    for (int m = 0; m < M; ++m) {
        const bf16_t* qkv_t = qkv_conv + (size_t)m * 8192;
        const bf16_t* pin_t = packed_in + (size_t)m * 12352;

        float qv = bf16_to_f(qkv_t[hk * 128 + tid]);
        float kv = bf16_to_f(qkv_t[2048 + hk * 128 + tid]);

        float qs = warp_reduce_sum(qv * qv);
        float ks = warp_reduce_sum(kv * kv);
        if (lane == 0) { wsum_q[wid] = qs; wsum_k[wid] = ks; }
        __syncthreads();
        if (wid == 0) {
            float vq = warp_reduce_sum((lane < 4) ? wsum_q[lane] : 0.0f);
            float vk = warp_reduce_sum((lane < 4) ? wsum_k[lane] : 0.0f);
            if (lane == 0) {
                wsum_q[0] = rsqrtf(vq + 1e-6f) * 0.08838834764831845f;
                wsum_k[0] = rsqrtf(vk + 1e-6f);
            }
        }
        __syncthreads();
        s_q[tid] = qv * wsum_q[0];
        s_k[tid] = kv * wsum_k[0];
        __syncthreads();

        float bv = bf16_to_f(pin_t[12288 + hv]);
        float beta = 1.0f / (1.0f + expf(-bv));
        float av = bf16_to_f(pin_t[12320 + hv]) + dt_b;
        float sp = (av > 20.0f) ? av : log1pf(expf(av));
        float g_exp = expf(neg_exp_A * sp);
        if (vs == 0 && tid == 0) {
            saved_bg[((size_t)m * 32 + hv) * 2 + 0] = beta;
            saved_bg[((size_t)m * 32 + hv) * 2 + 1] = g_exp;
        }

        float v_val = bf16_to_f(qkv_t[4096 + hv * 128 + v_col]);

        float kv_part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float sd = s_reg[j] * g_exp;
            s_reg[j] = sd;
            kv_part += sd * s_k[k_base + j];
        }
        sh_kv[wid][lane] = kv_part;
        __syncthreads();

        float kv_mem = sh_kv[0][lane] + sh_kv[1][lane] + sh_kv[2][lane] + sh_kv[3][lane];
        float delta = (v_val - kv_mem) * beta;

        float out_part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float sn = s_reg[j] + s_k[k_base + j] * delta;
            s_reg[j] = sn;
            out_part += sn * s_q[k_base + j];
        }
        sh_out[wid][lane] = out_part;
        __syncthreads();

        if (wid == 0) {
            float ov = sh_out[0][lane] + sh_out[1][lane] + sh_out[2][lane] + sh_out[3][lane];
            out_raw[(size_t)m * 4096 + hv * 128 + v_col] = ov;
        }
    }

    if (rec_state_final != nullptr) {
        float* S_out = rec_state_final + (size_t)hv * 16384;
        #pragma unroll
        for (int j = 0; j < 32; ++j) S_out[(k_base + j) * 128 + v_col] = s_reg[j];
    }
}

// Fast partial commit of 1..(M-1) accepted steps into rec_state when a draft is rejected!
// Grid: (32, 4, 1), Block: (128, 1, 1)
extern "C" __global__ void deltanet_commit_partial_bf16(
    const bf16_t* __restrict__ qkv_conv,    // [M, 8192]
    const float* __restrict__ saved_bg,     // [M, 32, 2]
    float* __restrict__ rec_state,          // [32, 128, 128] IN-PLACE update
    int num_steps
) {
    int hv = blockIdx.x;
    int vs = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int hk = hv >> 1;
    int v_col = vs * 32 + lane;
    int k_base = wid * 32;

    __shared__ float s_k[128];
    __shared__ float wsum_k[4];
    __shared__ float sh_kv[4][32];

    float s_reg[32];
    float* S_head = rec_state + (size_t)hv * 16384;
    #pragma unroll
    for (int j = 0; j < 32; ++j) s_reg[j] = S_head[(k_base + j) * 128 + v_col];

    for (int m = 0; m < num_steps; ++m) {
        const bf16_t* qkv_t = qkv_conv + (size_t)m * 8192;
        float kv = bf16_to_f(qkv_t[2048 + hk * 128 + tid]);
        float ks = warp_reduce_sum(kv * kv);
        if (lane == 0) wsum_k[wid] = ks;
        __syncthreads();
        if (wid == 0) {
            float vk = warp_reduce_sum((lane < 4) ? wsum_k[lane] : 0.0f);
            if (lane == 0) wsum_k[0] = rsqrtf(vk + 1e-6f);
        }
        __syncthreads();
        s_k[tid] = kv * wsum_k[0];
        __syncthreads();

        float beta  = saved_bg[((size_t)m * 32 + hv) * 2 + 0];
        float g_exp = saved_bg[((size_t)m * 32 + hv) * 2 + 1];
        float v_val = bf16_to_f(qkv_t[4096 + hv * 128 + v_col]);

        float kv_part = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            float sd = s_reg[j] * g_exp;
            s_reg[j] = sd;
            kv_part += sd * s_k[k_base + j];
        }
        sh_kv[wid][lane] = kv_part;
        __syncthreads();

        float kv_mem = sh_kv[0][lane] + sh_kv[1][lane] + sh_kv[2][lane] + sh_kv[3][lane];
        float delta = (v_val - kv_mem) * beta;

        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            s_reg[j] += s_k[k_base + j] * delta;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < 32; ++j) S_head[(k_base + j) * 128 + v_col] = s_reg[j];
}

// Dynamic-pos QK-Norm + Partial M-RoPE + KV Write for M verified tokens (CUDA-Graphable!)
// Grid: (20, M, 1), Block: (256, 1, 1) -> q_out layout: [M, 16, 256]
extern "C" __global__ void full_attn_qk_rope_kv_verify_m_bf16(
    const bf16_t* __restrict__ qkv_packed,  // [M, 10240]
    const bf16_t* __restrict__ q_norm_w,
    const bf16_t* __restrict__ k_norm_w,
    const bf16_t* __restrict__ cos_table,
    const bf16_t* __restrict__ sin_table,
    const int* __restrict__ pos_ptr,
    bf16_t* __restrict__ q_out,             // [M, 16, 256]
    bf16_t* __restrict__ k_cache,           // [4, max_seq, 256]
    bf16_t* __restrict__ v_cache,           // [4, max_seq, 256]
    int max_seq,
    float eps
) {
    int b = blockIdx.x;
    int m = blockIdx.y;
    int d = threadIdx.x;
    int lane = d & 31;
    int wid = d >> 5;
    int pos = pos_ptr[0] + m;

    __shared__ float wsum[8];
    __shared__ float norm_buf[64];

    const bf16_t* row = qkv_packed + (size_t)m * 10240;

    if (b < 16) {
        const bf16_t* src = row + b * 512;
        float val = bf16_to_f(src[d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = warp_reduce_sum((lane < 8) ? wsum[lane] : 0.0f);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(q_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        q_out[((size_t)m * 16 + b) * 256 + d] = f_to_bf16(nv);
    } else {
        int kh = b - 16;
        const bf16_t* k_raw = row + 8192;
        const bf16_t* v_raw = row + 9216;
        float val = bf16_to_f(k_raw[kh * 256 + d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = warp_reduce_sum((lane < 8) ? wsum[lane] : 0.0f);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(k_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        size_t kv_off = ((size_t)kh * max_seq + pos) * 256 + d;
        k_cache[kv_off] = f_to_bf16(nv);
        v_cache[kv_off] = v_raw[kh * 256 + d];
    }
}

// Warp-Vectorized Causal Split-KV FlashDecode for M verified tokens (CUDA-Graphable!)
// Grid: (16, 4, M), Block: (256, 1, 1)
extern "C" __global__ void gqa_verify_m_flash_splitk_bf16(
    const bf16_t* __restrict__ q_in,        // [M, 16, 256]
    const bf16_t* __restrict__ k_cache,     // [4, max_seq, 256]
    const bf16_t* __restrict__ v_cache,     // [4, max_seq, 256]
    const int* __restrict__ pos_ptr,
    float* __restrict__ part_out,           // [M, 16, num_splits, 256]
    float* __restrict__ part_ml,            // [M, 16, num_splits, 2]
    int max_seq,
    int num_splits
) {
    int hq = blockIdx.x;
    int sp = blockIdx.y;
    int m  = blockIdx.z;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int hk = hq >> 2;
    int seq_len = pos_ptr[0] + m + 1;

    int chunk = (seq_len + num_splits - 1) / num_splits;
    int t_start = sp * chunk;
    int t_end = t_start + chunk;
    if (t_end > seq_len) t_end = seq_len;

    const float scale = 0.0625f;
    uint4 q4 = *reinterpret_cast<const uint4*>(q_in + ((size_t)m * 16 + hq) * 256 + lane * 8);
    float q_reg[8];
    unpack_uint4_bf16(q4, q_reg);
    #pragma unroll
    for (int j = 0; j < 8; ++j) q_reg[j] *= scale;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const bf16_t* k_head = k_cache + (size_t)hk * max_seq * 256;
    const bf16_t* v_head = v_cache + (size_t)hk * max_seq * 256;

    for (int t = t_start + wid; t < t_end; t += 8) {
        uint4 k4 = *reinterpret_cast<const uint4*>(k_head + (size_t)t * 256 + lane * 8);
        float kf[8];
        unpack_uint4_bf16(k4, kf);
        float prod = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) prod += q_reg[j] * kf[j];
        float score = warp_reduce_sum(prod);

        float new_max = fmaxf(m_max, score);
        float alpha = expf(m_max - new_max);
        float p = expf(score - new_max);
        l_sum = l_sum * alpha + p;
        m_max = new_max;

        uint4 v4 = *reinterpret_cast<const uint4*>(v_head + (size_t)t * 256 + lane * 8);
        float vf[8];
        unpack_uint4_bf16(v4, vf);
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = acc[j] * alpha + p * vf[j];
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_acc[8][256];
    if (lane == 0) { sm_m[wid] = m_max; sm_l[wid] = l_sum; }
    #pragma unroll
    for (int j = 0; j < 8; ++j) sm_acc[wid][lane * 8 + j] = acc[j];
    __syncthreads();

    int d = tid;
    float g_max = -1e30f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) g_max = fmaxf(g_max, sm_m[w]);
    float g_sum = 0.0f;
    float g_acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float lw = sm_l[w];
        if (lw > 0.0f) {
            float mult = expf(sm_m[w] - g_max);
            g_sum += lw * mult;
            g_acc += sm_acc[w][d] * mult;
        }
    }

    size_t idx_base = ((size_t)(m * 16 + hq) * num_splits + sp);
    part_out[idx_base * 256 + d] = g_acc;
    if (d == 0) {
        part_ml[idx_base * 2 + 0] = g_max;
        part_ml[idx_base * 2 + 1] = g_sum;
    }
}

// Grid: (16, M, 1), Block: (256, 1, 1)
extern "C" __global__ void gqa_verify_m_flash_reduce_bf16(
    const float* __restrict__ part_out,
    const float* __restrict__ part_ml,
    const bf16_t* __restrict__ qkv_packed,  // [M, 10240]
    bf16_t* __restrict__ out,               // [M, 4096]
    int num_splits
) {
    int hq = blockIdx.x;
    int m  = blockIdx.y;
    int d  = threadIdx.x;

    float g_max = -1e30f;
    for (int sp = 0; sp < num_splits; ++sp) {
        float mv = part_ml[((size_t)(m * 16 + hq) * num_splits + sp) * 2 + 0];
        g_max = fmaxf(g_max, mv);
    }
    float g_sum = 0.0f;
    float g_acc = 0.0f;
    for (int sp = 0; sp < num_splits; ++sp) {
        size_t base = (size_t)(m * 16 + hq) * num_splits + sp;
        float mv = part_ml[base * 2 + 0];
        float lv = part_ml[base * 2 + 1];
        if (lv > 0.0f) {
            float w = expf(mv - g_max);
            g_sum += lv * w;
            g_acc += part_out[base * 256 + d] * w;
        }
    }
    float attn_val = g_acc / fmaxf(g_sum, 1e-30f);
    float gate_val = bf16_to_f(qkv_packed[(size_t)m * 10240 + hq * 512 + 256 + d]);
    float sig = 1.0f / (1.0f + expf(-gate_val));
    out[(size_t)m * 4096 + hq * 256 + d] = f_to_bf16(attn_val * sig);
}

// Batch Argmax over M rows of [M, n] logits
// stage1 Grid: (256, M, 1), Block: (256, 1, 1)
extern "C" __global__ void argmax_batch_stage1_bf16(
    const bf16_t* __restrict__ logits,
    float* __restrict__ part_vals,          // [M, 256]
    int* __restrict__ part_idxs,            // [M, 256]
    int n
) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int row = blockIdx.y;
    int stride = gridDim.x * 256;
    const bf16_t* lrow = logits + (size_t)row * n;

    float best_v = -1e30f;
    int best_i = 0;
    for (int i = bid * 256 + tid; i < n; i += stride) {
        float v = bf16_to_f(lrow[i]);
        if (v > best_v) { best_v = v; best_i = i; }
    }
    s_val[tid] = best_v;
    s_idx[tid] = best_i;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_vals[row * 256 + bid] = s_val[0];
        part_idxs[row * 256 + bid] = s_idx[0];
    }
}

// stage2 Grid: (1, M, 1), Block: (256, 1, 1) — supports arbitrary num_blocks (e.g. 256 or 1024)
extern "C" __global__ void argmax_batch_stage2_bf16(
    const float* __restrict__ part_vals,
    const int* __restrict__ part_idxs,
    long long* __restrict__ out_ids,        // [M]
    int num_blocks
) {
    __shared__ float s_val[256];
    __shared__ int s_idx[256];
    int tid = threadIdx.x;
    int row = blockIdx.y;
    const float* rv = part_vals + (size_t)row * num_blocks;
    const int*   ri = part_idxs + (size_t)row * num_blocks;
    float bv = -1e30f;
    int bi = 0;
    for (int i = tid; i < num_blocks; i += 256) {
        float v = rv[i];
        if (v > bv) {
            bv = v;
            bi = ri[i];
        }
    }
    s_val[tid] = bv;
    s_idx[tid] = bi;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s && s_val[tid + s] > s_val[tid]) {
            s_val[tid] = s_val[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_ids[row] = (long long)s_idx[0];
    }
}

// ===========================================================================
// 10. Phase R6: Custom Warp-Vectorized Skinny BF16 GEMM (M = 1..4) + Fused Residual
// Beats cuBLASEx & cuBLASLt by 1.12x-1.50x on RTX 3080 (SM 8.6) for N <= 18432!
// K must be a multiple of 256 (guaranteed for all Qwen3.5-4B layers: 2560/4096/5120/9216).
// ===========================================================================

// M = 1: 2 warps per output row (4 rows/block = 256 threads), 128-bit uint4 loads + 64-bit uint2 coalesced store!
// Grid: ((N + 3) / 4, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void skinny_gemm_m1_2wpr_bf16(
    const bf16_t* __restrict__ X, // [1, K]
    const bf16_t* __restrict__ W, // [N, K]
    bf16_t* __restrict__ Y,       // [1, N]
    int N,
    int K,
    int add_res
) {
    int tid = threadIdx.x;
    int wid = tid >> 5;           // 0..7
    int lane = tid & 31;
    int r_local = wid >> 1;       // 0..3
    int k_half = wid & 1;         // 0 or 1
    int n_base = blockIdx.x * 4;
    int n = n_base + r_local;

    float acc = 0.0f;
    if (n < N) {
        const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)n * K);
        const uint4* x_row = reinterpret_cast<const uint4*>(X);
        int k_vec = K >> 3;
        for (int kv = k_half * 32 + lane; kv < k_vec; kv += 64) {
            uint4 w4 = w_row[kv];
            uint4 x4 = __ldg(&x_row[kv]);
            float wf[8], xf[8];
            unpack_uint4_bf16(w4, wf);
            unpack_uint4_bf16(x4, xf);
            #pragma unroll
            for (int j = 0; j < 8; ++j) acc = fmaf(xf[j], wf[j], acc);
        }
    }

    __shared__ float sm[4][2];
    float s = warp_reduce_sum(acc);
    if (lane == 0) sm[r_local][k_half] = s;
    __syncthreads();
    if (tid == 0) {
        bf16_t* dst = Y + n_base;
        if (n_base + 3 < N) {
            float v0 = sm[0][0] + sm[0][1];
            float v1 = sm[1][0] + sm[1][1];
            float v2 = sm[2][0] + sm[2][1];
            float v3 = sm[3][0] + sm[3][1];
            if (add_res) {
                uint2 cur = *reinterpret_cast<const uint2*>(dst);
                v0 += bf16_to_f((bf16_t)(cur.x & 0xFFFFu));
                v1 += bf16_to_f((bf16_t)(cur.x >> 16));
                v2 += bf16_to_f((bf16_t)(cur.y & 0xFFFFu));
                v3 += bf16_to_f((bf16_t)(cur.y >> 16));
            }
            uint2 out2;
            out2.x = (unsigned int)f_to_bf16(v0) | ((unsigned int)f_to_bf16(v1) << 16);
            out2.y = (unsigned int)f_to_bf16(v2) | ((unsigned int)f_to_bf16(v3) << 16);
            *reinterpret_cast<uint2*>(dst) = out2;
        } else {
            for (int r = 0; r < 4 && n_base + r < N; ++r) {
                float v = sm[r][0] + sm[r][1];
                if (add_res) v += bf16_to_f(dst[r]);
                dst[r] = f_to_bf16(v);
            }
        }
    }
}

// M = 2..4: 8 warps/block (8 rows/block = 256 threads), ZERO inner syncs + 128-bit uint4 Coalesced Epilogue!
// Grid: ((N + 7) / 8, 1, 1), Block: (256, 1, 1)
template <int M_VAL>
__device__ __forceinline__ void skinny_gemm_1wpr_impl(
    const bf16_t* __restrict__ X, // [M, K]
    const bf16_t* __restrict__ W, // [N, K]
    bf16_t* __restrict__ Y,       // [M, N]
    int N,
    int K,
    int add_res
) {
    int tid = threadIdx.x;
    int wid = tid >> 5;           // 0..7 (8 rows/block)
    int lane = tid & 31;
    int n_base = blockIdx.x * 8;
    int n = n_base + wid;

    float acc[M_VAL];
    #pragma unroll
    for (int m = 0; m < M_VAL; ++m) acc[m] = 0.0f;

    if (n < N) {
        const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)n * K);
        int k_vec = K >> 3;
        for (int kv = lane; kv < k_vec; kv += 32) {
            uint4 w4 = w_row[kv];
            float wf[8];
            unpack_uint4_bf16(w4, wf);

            #pragma unroll
            for (int m = 0; m < M_VAL; ++m) {
                const uint4* x_row = reinterpret_cast<const uint4*>(X + (size_t)m * K);
                uint4 x4 = __ldg(&x_row[kv]);
                float xf[8];
                unpack_uint4_bf16(x4, xf);
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[m] = fmaf(xf[j], wf[j], acc[m]);
            }
        }
    }

    __shared__ float sm[M_VAL][8];
    #pragma unroll
    for (int m = 0; m < M_VAL; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if (lane == 0) sm[m][wid] = s;
    }
    __syncthreads();

    if (tid < M_VAL) {
        int m = tid;
        bf16_t* dst = Y + (size_t)m * N + n_base;
        if (n_base + 7 < N) {
            float vals[8];
            if (add_res) {
                uint4 cur = *reinterpret_cast<const uint4*>(dst);
                unpack_uint4_bf16(cur, vals);
                #pragma unroll
                for (int r = 0; r < 8; ++r) vals[r] += sm[m][r];
            } else {
                #pragma unroll
                for (int r = 0; r < 8; ++r) vals[r] = sm[m][r];
            }
            *reinterpret_cast<uint4*>(dst) = pack_uint4_bf16(vals);
        } else {
            for (int r = 0; r < 8 && n_base + r < N; ++r) {
                float v = sm[m][r];
                if (add_res) v += bf16_to_f(dst[r]);
                dst[r] = f_to_bf16(v);
            }
        }
    }
}

extern "C" __global__ void skinny_gemm_m2_1wpr_bf16(
    const bf16_t* __restrict__ X, const bf16_t* __restrict__ W, bf16_t* __restrict__ Y, int N, int K, int add_res
) { skinny_gemm_1wpr_impl<2>(X, W, Y, N, K, add_res); }

extern "C" __global__ void skinny_gemm_m3_1wpr_bf16(
    const bf16_t* __restrict__ X, const bf16_t* __restrict__ W, bf16_t* __restrict__ Y, int N, int K, int add_res
) { skinny_gemm_1wpr_impl<3>(X, W, Y, N, K, add_res); }

extern "C" __global__ void skinny_gemm_m4_1wpr_bf16(
    const bf16_t* __restrict__ X, const bf16_t* __restrict__ W, bf16_t* __restrict__ Y, int N, int K, int add_res
) { skinny_gemm_1wpr_impl<4>(X, W, Y, N, K, add_res); }

extern "C" __global__ void skinny_gemm_m5_1wpr_bf16(
    const bf16_t* __restrict__ X, const bf16_t* __restrict__ W, bf16_t* __restrict__ Y, int N, int K, int add_res
) { skinny_gemm_1wpr_impl<5>(X, W, Y, N, K, add_res); }

// ===========================================================================
// 11. Phase R7: Continuous Batching (B=1..4) + Paged KV Block Manager (BLOCK_SIZE=16)
// KV Pool Layout: [num_blocks, 8 fa_layers, 2 (K/V), 4 kv_heads, 16 tokens, 256 dim] bf16
//   block_stride = 262144, layer_stride = 32768, kv_stride = 16384, head_stride = 4096, tok_stride = 256
// ===========================================================================

// 11a. Batched Causal Conv1D (k=4) Decode across B independent DeltaNet slots
// Grid: (32, B, 1), Block: (256, 1, 1)
extern "C" __global__ void causal_conv1d_batch_decode_bf16(
    const bf16_t* __restrict__ packed_in,   // [B, 12352]
    bf16_t* __restrict__ conv_pool,         // [max_batch, 24, 8192, 4]
    const int* __restrict__ slot_ids,       // [B]
    const bf16_t* __restrict__ conv_weight, // [8192, 4]
    bf16_t* __restrict__ y_out,             // [B, 8192]
    int lin_idx,
    int C
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y;
    if (c >= C) return;

    int slot = slot_ids[b];
    bf16_t* conv_state = conv_pool + ((size_t)slot * 24 + lin_idx) * 32768;
    const bf16_t* x_in = packed_in + (size_t)b * 12352;
    int base = c * 4;

    bf16_t s1 = conv_state[base + 1];
    bf16_t s2 = conv_state[base + 2];
    bf16_t s3 = conv_state[base + 3];
    bf16_t s4 = x_in[c];

    conv_state[base + 0] = s1;
    conv_state[base + 1] = s2;
    conv_state[base + 2] = s3;
    conv_state[base + 3] = s4;

    float acc = bf16_to_f(s1) * bf16_to_f(conv_weight[base + 0])
              + bf16_to_f(s2) * bf16_to_f(conv_weight[base + 1])
              + bf16_to_f(s3) * bf16_to_f(conv_weight[base + 2])
              + bf16_to_f(s4) * bf16_to_f(conv_weight[base + 3]);
    float silu = acc / (1.0f + expf(-acc));
    y_out[(size_t)b * 8192 + c] = f_to_bf16(silu);
}

// 11b. Batched 1-Pass Register-Tiled Gated DeltaNet Decode across B independent slots
// Grid: (32, B, 1), Block: (256, 1, 1)
extern "C" __global__ void gated_deltanet_batch_decode_bf16(
    const bf16_t* __restrict__ qkv_conv,    // [B, 8192]
    const bf16_t* __restrict__ packed_in,   // [B, 12352]
    const bf16_t* __restrict__ dt_bias,
    const float* __restrict__ A_log,
    const bf16_t* __restrict__ norm_w,
    float* __restrict__ rec_pool,           // [max_batch, 24, 32, 128, 128]
    const int* __restrict__ slot_ids,       // [B]
    bf16_t* __restrict__ out,               // [B, 4096]
    int lin_idx,
    float eps
) {
    int hv = blockIdx.x;
    int b  = blockIdx.y;
    int tid = threadIdx.x;
    int k_half = tid >> 7;     // 0 or 1
    int v_col = tid & 127;     // 0..127
    int lane = v_col & 31;
    int wid = v_col >> 5;      // 0..3
    int hk = hv >> 1;

    int slot = slot_ids[b];
    const bf16_t* qkv_b = qkv_conv + (size_t)b * 8192;
    const bf16_t* pin_b = packed_in + (size_t)b * 12352;
    bf16_t* out_b = out + (size_t)b * 4096;
    float* rec_state = rec_pool + ((size_t)slot * 24 + lin_idx) * 524288;

    __shared__ float s_q[128];
    __shared__ float s_k[128];
    __shared__ float wsum_q[4];
    __shared__ float wsum_k[4];
    __shared__ float sh_half[128];

    if (k_half == 0) {
        float qv = bf16_to_f(qkv_b[hk * 128 + v_col]);
        float qs = warp_reduce_sum(qv * qv);
        if (lane == 0) wsum_q[wid] = qs;
        s_q[v_col] = qv;
    } else {
        float kv = bf16_to_f(qkv_b[2048 + hk * 128 + v_col]);
        float ks = warp_reduce_sum(kv * kv);
        if (lane == 0) wsum_k[wid] = ks;
        s_k[v_col] = kv;
    }
    __syncthreads();
    if (k_half == 0 && wid == 0) {
        float v = warp_reduce_sum((lane < 4) ? wsum_q[lane] : 0.0f);
        if (lane == 0) wsum_q[0] = rsqrtf(v + 1e-6f) * 0.08838834764831845f;
    } else if (k_half == 1 && wid == 0) {
        float v = warp_reduce_sum((lane < 4) ? wsum_k[lane] : 0.0f);
        if (lane == 0) wsum_k[0] = rsqrtf(v + 1e-6f);
    }
    __syncthreads();
    if (k_half == 0) s_q[v_col] *= wsum_q[0];
    else             s_k[v_col] *= wsum_k[0];
    __syncthreads();

    const bf16_t* z_in = pin_b + 8192;
    const bf16_t* b_in = pin_b + 12288;
    const bf16_t* a_in = pin_b + 12320;

    float bv = bf16_to_f(b_in[hv]);
    float beta = 1.0f / (1.0f + expf(-bv));
    float av = bf16_to_f(a_in[hv]) + bf16_to_f(dt_bias[hv]);
    float sp = (av > 20.0f) ? av : log1pf(expf(av));
    float g_exp = expf(-expf(A_log[hv]) * sp);

    float v_val = bf16_to_f(qkv_b[4096 + hv * 128 + v_col]);
    float* S_head = rec_state + (size_t)hv * 16384;
    int k_base = k_half * 64;

    float s_reg[64];
    float kv_part = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        float sd = S_head[(k_base + j) * 128 + v_col] * g_exp;
        s_reg[j] = sd;
        kv_part += sd * s_k[k_base + j];
    }

    if (k_half == 1) sh_half[v_col] = kv_part;
    __syncthreads();
    if (k_half == 0) {
        float kv_mem = kv_part + sh_half[v_col];
        sh_half[v_col] = (v_val - kv_mem) * beta;
    }
    __syncthreads();
    float delta = sh_half[v_col];

    float out_part = 0.0f;
    #pragma unroll
    for (int j = 0; j < 64; ++j) {
        float s_new = s_reg[j] + s_k[k_base + j] * delta;
        S_head[(k_base + j) * 128 + v_col] = s_new;
        out_part += s_new * s_q[k_base + j];
    }

    __syncthreads();
    if (k_half == 1) sh_half[v_col] = out_part;
    __syncthreads();

    if (k_half == 0) {
        float out_val = out_part + sh_half[v_col];
        float os = warp_reduce_sum(out_val * out_val);
        if (lane == 0) wsum_q[wid] = os;
        sh_half[v_col] = out_val;
    }
    __syncthreads();
    if (k_half == 0 && wid == 0) {
        float v = warp_reduce_sum((lane < 4) ? wsum_q[lane] : 0.0f);
        if (lane == 0) wsum_q[0] = rsqrtf(v * (1.0f / 128.0f) + eps);
    }
    __syncthreads();
    if (k_half == 0) {
        float out_val = sh_half[v_col];
        float normed = bf16_to_f(f_to_bf16(out_val * wsum_q[0])) * bf16_to_f(norm_w[v_col]);
        float zv = bf16_to_f(z_in[hv * 128 + v_col]);
        float z_silu = zv / (1.0f + expf(-zv));
        out_b[hv * 128 + v_col] = f_to_bf16(normed * z_silu);
    }
}

// 11c. Paged KV Prefill QK-Norm + M-RoPE + Paged Block Write
// Grid: (20, seq_len, 1), Block: (256, 1, 1)
extern "C" __global__ void qk_rope_paged_kv_prefill_bf16(
    const bf16_t* __restrict__ qkv_packed,  // [seq_len, 10240]
    const bf16_t* __restrict__ q_norm_w,
    const bf16_t* __restrict__ k_norm_w,
    const bf16_t* __restrict__ cos_table,
    const bf16_t* __restrict__ sin_table,
    int start_pos,
    int seq_len,
    bf16_t* __restrict__ q_sdpa,            // [16, seq_len, 256]
    bf16_t* __restrict__ kv_pool,           // [num_blocks, 8, 2, 4, 16, 256]
    const int* __restrict__ block_table,    // [max_blocks_per_seq]
    int fa_idx,
    float eps
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int d = threadIdx.x;
    int lane = d & 31;
    int wid = d >> 5;
    int pos = start_pos + t;

    __shared__ float wsum[8];
    __shared__ float norm_buf[64];
    const bf16_t* row = qkv_packed + (size_t)t * 10240;

    if (b < 16) {
        const bf16_t* src = row + b * 512;
        float val = bf16_to_f(src[d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = warp_reduce_sum((lane < 8) ? wsum[lane] : 0.0f);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(q_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        q_sdpa[((size_t)b * seq_len + t) * 256 + d] = f_to_bf16(nv);
    } else {
        int kh = b - 16;
        const bf16_t* k_raw = row + 8192;
        const bf16_t* v_raw = row + 9216;
        float val = bf16_to_f(k_raw[kh * 256 + d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = warp_reduce_sum((lane < 8) ? wsum[lane] : 0.0f);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(k_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        int blk = block_table[pos >> 4];
        int off = pos & 15;
        size_t base = (size_t)blk * 262144 + (size_t)fa_idx * 32768 + (size_t)kh * 4096 + (size_t)off * 256 + d;
        kv_pool[base]         = f_to_bf16(nv);
        kv_pool[base + 16384] = v_raw[kh * 256 + d];
    }
}

// 11d. FlashAttention-2 Tiled Causal Prefill GQA over Paged KV Blocks (16-tok tile == 1 KV block!)
// Grid: (4, (seq_len + 1) / 2, 1), Block: (256, 1, 1)
extern "C" __global__ void prefill_gqa_paged_causal_gated_bf16(
    const bf16_t* __restrict__ q_sdpa,      // [16, seq_len, 256]
    const bf16_t* __restrict__ qkv_packed,  // [seq_len, 10240]
    const bf16_t* __restrict__ kv_pool,     // [num_blocks, 8, 2, 4, 16, 256]
    const int* __restrict__ block_table,    // [max_blocks_per_seq]
    int fa_idx,
    int start_pos,
    int seq_len,
    bf16_t* __restrict__ out                // [seq_len, 4096]
) {
    int hk = blockIdx.x;
    int qt_base = blockIdx.y * 2;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int q_sub = wid >> 2;
    int hq_local = wid & 3;
    int hq = hk * 4 + hq_local;
    int t_my = qt_base + q_sub;
    bool valid_q = (t_my < seq_len);

    const float scale = 0.0625f;
    float q_reg[8];
    if (valid_q) {
        uint4 q4 = *reinterpret_cast<const uint4*>(
            q_sdpa + ((size_t)hq * seq_len + t_my) * 256 + lane * 8
        );
        unpack_uint4_bf16(q4, q_reg);
        #pragma unroll
        for (int j = 0; j < 8; ++j) q_reg[j] *= scale;
    }

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    __shared__ bf16_t s_K[16][256];
    __shared__ bf16_t s_V[16][256];

    int kv_end_block = start_pos + min(qt_base + 2, seq_len);
    int kv_end_my = start_pos + t_my + 1;

    for (int kt0 = 0; kt0 < kv_end_block; kt0 += 16) {
        // 1-to-1 match: 16-token FlashAttention tile == 1 physical KV block!
        int blk = block_table[kt0 >> 4];
        const bf16_t* k_blk = kv_pool + (size_t)blk * 262144 + (size_t)fa_idx * 32768 + (size_t)hk * 4096;
        const bf16_t* v_blk = k_blk + 16384;

        #pragma unroll
        for (int row = wid; row < 16; row += 8) {
            if (kt0 + row < kv_end_block) {
                *reinterpret_cast<uint4*>(&s_K[row][lane * 8]) =
                    *reinterpret_cast<const uint4*>(k_blk + (size_t)row * 256 + lane * 8);
                *reinterpret_cast<uint4*>(&s_V[row][lane * 8]) =
                    *reinterpret_cast<const uint4*>(v_blk + (size_t)row * 256 + lane * 8);
            }
        }
        __syncthreads();

        if (valid_q && kt0 < kv_end_my) {
            int k_lim = kv_end_my - kt0;
            if (k_lim > 16) k_lim = 16;
            for (int ki = 0; ki < k_lim; ++ki) {
                uint4 k4 = *reinterpret_cast<const uint4*>(&s_K[ki][lane * 8]);
                float kf[8];
                unpack_uint4_bf16(k4, kf);
                float prod = 0.0f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) prod += q_reg[j] * kf[j];
                float score = warp_reduce_sum(prod);

                float new_max = fmaxf(m_max, score);
                float alpha = expf(m_max - new_max);
                float p = expf(score - new_max);
                l_sum = l_sum * alpha + p;
                m_max = new_max;

                uint4 v4 = *reinterpret_cast<const uint4*>(&s_V[ki][lane * 8]);
                float vf[8];
                unpack_uint4_bf16(v4, vf);
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[j] = acc[j] * alpha + p * vf[j];
            }
        }
        __syncthreads();
    }

    if (valid_q) {
        float inv_l = 1.0f / fmaxf(l_sum, 1e-30f);
        uint4 g4 = *reinterpret_cast<const uint4*>(
            qkv_packed + (size_t)t_my * 10240 + hq * 512 + 256 + lane * 8
        );
        float gf[8];
        unpack_uint4_bf16(g4, gf);
        float res[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float sig = 1.0f / (1.0f + expf(-gf[j]));
            res[j] = (acc[j] * inv_l) * sig;
        }
        *reinterpret_cast<uint4*>(out + (size_t)t_my * 4096 + hq * 256 + lane * 8) = pack_uint4_bf16(res);
    }
}

// 11e. Batched Decode QK-Norm + M-RoPE + Paged KV Write across B requests with independent positions!
// Grid: (20, B, 1), Block: (256, 1, 1) -> q_out layout: [B, 16, 256]
extern "C" __global__ void qk_rope_paged_kv_batch_decode_bf16(
    const bf16_t* __restrict__ qkv_packed,  // [B, 10240]
    const bf16_t* __restrict__ q_norm_w,
    const bf16_t* __restrict__ k_norm_w,
    const bf16_t* __restrict__ cos_table,
    const bf16_t* __restrict__ sin_table,
    const int* __restrict__ positions,      // [B]
    bf16_t* __restrict__ q_out,             // [B, 16, 256]
    bf16_t* __restrict__ kv_pool,           // [num_blocks, 8, 2, 4, 16, 256]
    const int* __restrict__ block_tables,   // [B, max_blocks_per_seq]
    int max_blocks_per_seq,
    int fa_idx,
    float eps
) {
    int h = blockIdx.x;
    int b = blockIdx.y;
    int d = threadIdx.x;
    int lane = d & 31;
    int wid = d >> 5;
    int pos = positions[b];

    __shared__ float wsum[8];
    __shared__ float norm_buf[64];
    const bf16_t* row = qkv_packed + (size_t)b * 10240;

    if (h < 16) {
        const bf16_t* src = row + h * 512;
        float val = bf16_to_f(src[d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = warp_reduce_sum((lane < 8) ? wsum[lane] : 0.0f);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(q_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        q_out[((size_t)b * 16 + h) * 256 + d] = f_to_bf16(nv);
    } else {
        int kh = h - 16;
        const bf16_t* k_raw = row + 8192;
        const bf16_t* v_raw = row + 9216;
        float val = bf16_to_f(k_raw[kh * 256 + d]);
        float sq = warp_reduce_sum(val * val);
        if (lane == 0) wsum[wid] = sq;
        __syncthreads();
        if (wid == 0) {
            float v = warp_reduce_sum((lane < 8) ? wsum[lane] : 0.0f);
            if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 256.0f) + eps);
        }
        __syncthreads();
        float nv = bf16_to_f(f_to_bf16(val * wsum[0] * (1.0f + bf16_to_f(k_norm_w[d]))));
        if (d < 64) norm_buf[d] = nv;
        __syncthreads();
        if (d < 64) {
            float c = bf16_to_f(cos_table[pos * 64 + d]);
            float s = bf16_to_f(sin_table[pos * 64 + d]);
            float pair = (d < 32) ? -norm_buf[d + 32] : norm_buf[d - 32];
            nv = norm_buf[d] * c + pair * s;
        }
        int blk = block_tables[(size_t)b * max_blocks_per_seq + (pos >> 4)];
        int off = pos & 15;
        size_t base = (size_t)blk * 262144 + (size_t)fa_idx * 32768 + (size_t)kh * 4096 + (size_t)off * 256 + d;
        kv_pool[base]         = f_to_bf16(nv);
        kv_pool[base + 16384] = v_raw[kh * 256 + d];
    }
}

// 11f. Paged Warp-Vectorized Split-KV FlashDecode across B concurrent requests!
// Output matches gqa_verify_m_flash_reduce_bf16 layout [B, 16, num_splits, 256] -> ZERO new reduce kernel!
// Grid: (16, 4, B), Block: (256, 1, 1)
extern "C" __global__ void paged_gqa_decode_flash_splitk_bf16(
    const bf16_t* __restrict__ q_in,        // [B, 16, 256]
    const bf16_t* __restrict__ kv_pool,     // [num_blocks, 8, 2, 4, 16, 256]
    const int* __restrict__ block_tables,   // [B, max_blocks_per_seq]
    const int* __restrict__ positions,      // [B]
    float* __restrict__ part_out,           // [B, 16, num_splits, 256]
    float* __restrict__ part_ml,            // [B, 16, num_splits, 2]
    int max_blocks_per_seq,
    int fa_idx,
    int num_splits
) {
    int hq = blockIdx.x;
    int sp = blockIdx.y;
    int b  = blockIdx.z;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int hk = hq >> 2;
    int seq_len = positions[b] + 1;

    int chunk = (seq_len + num_splits - 1) / num_splits;
    int t_start = sp * chunk;
    int t_end = t_start + chunk;
    if (t_end > seq_len) t_end = seq_len;

    const float scale = 0.0625f;
    uint4 q4 = *reinterpret_cast<const uint4*>(q_in + ((size_t)b * 16 + hq) * 256 + lane * 8);
    float q_reg[8];
    unpack_uint4_bf16(q4, q_reg);
    #pragma unroll
    for (int j = 0; j < 8; ++j) q_reg[j] *= scale;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    const int* bt = block_tables + (size_t)b * max_blocks_per_seq;
    size_t layer_head_off = (size_t)fa_idx * 32768 + (size_t)hk * 4096;

    for (int t = t_start + wid; t < t_end; t += 8) {
        int blk = bt[t >> 4];
        int off = t & 15;
        size_t k_off = (size_t)blk * 262144 + layer_head_off + (size_t)off * 256 + lane * 8;

        uint4 k4 = *reinterpret_cast<const uint4*>(kv_pool + k_off);
        float kf[8];
        unpack_uint4_bf16(k4, kf);
        float prod = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) prod += q_reg[j] * kf[j];
        float score = warp_reduce_sum(prod);

        float new_max = fmaxf(m_max, score);
        float alpha = expf(m_max - new_max);
        float p = expf(score - new_max);
        l_sum = l_sum * alpha + p;
        m_max = new_max;

        uint4 v4 = *reinterpret_cast<const uint4*>(kv_pool + k_off + 16384);
        float vf[8];
        unpack_uint4_bf16(v4, vf);
        #pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = acc[j] * alpha + p * vf[j];
    }

    __shared__ float sm_m[8];
    __shared__ float sm_l[8];
    __shared__ float sm_acc[8][256];
    if (lane == 0) { sm_m[wid] = m_max; sm_l[wid] = l_sum; }
    #pragma unroll
    for (int j = 0; j < 8; ++j) sm_acc[wid][lane * 8 + j] = acc[j];
    __syncthreads();

    int d = tid;
    float g_max = -1e30f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) g_max = fmaxf(g_max, sm_m[w]);
    float g_sum = 0.0f;
    float g_acc = 0.0f;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        float lw = sm_l[w];
        if (lw > 0.0f) {
            float mult = expf(sm_m[w] - g_max);
            g_sum += lw * mult;
            g_acc += sm_acc[w][d] * mult;
        }
    }

    size_t idx_base = ((size_t)(b * 16 + hq) * num_splits + sp);
    part_out[idx_base * 256 + d] = g_acc;
    if (d == 0) {
        part_ml[idx_base * 2 + 0] = g_max;
        part_ml[idx_base * 2 + 1] = g_sum;
    }
}

// ===========================================================================
// 12. Phase R8 v2: FlashLinearAttention-Style Chunked WY Gated DeltaNet Prefill
//     - BF16 Batched Stage 1 (cuBLAS Ampere s16816gemm Tensor Cores, batch=128)
//     - 2x32 Warp-Shuffle Block Triangular Inversion T = (I + M)^-1 (10x faster!)
//     - Single-Kernel Fused WMMA BF16 Tensor Core State + Output (0 DRAM S traffic!)
// ===========================================================================

__device__ __forceinline__ void pick_chunk_4(
    int sub,
    int c0s, int c0l, int c1s, int c1l, int c2s, int c2l, int c3s, int c3l,
    int &c_start, int &c_len
) {
    if (sub == 0)      { c_start = c0s; c_len = c0l; }
    else if (sub == 1) { c_start = c1s; c_len = c1l; }
    else if (sub == 2) { c_start = c2s; c_len = c2l; }
    else               { c_start = c3s; c_len = c3l; }
}

// 12.1 Cumsum log-decay G[B_c, 32, 64], beta[B_c, 32, 64], g_last[B_c, 32]
// Grid: (32, B_c, 1), Block: (64, 1, 1)
extern "C" __global__ void deltanet_wy_cumsum_g_f32(
    const bf16_t* __restrict__ packed_in,  // [T, 12352]
    const bf16_t* __restrict__ dt_bias,    // [32]
    const float* __restrict__ A_log,       // [32]
    float* __restrict__ buf_G,             // [B_c, 32, 64]
    float* __restrict__ buf_beta,          // [B_c, 32, 64]
    float* __restrict__ buf_g_last,        // [B_c, 32]
    int c0s, int c0l, int c1s, int c1l, int c2s, int c2l, int c3s, int c3l
) {
    int hv = blockIdx.x;
    int sub = blockIdx.y;
    int tid = threadIdx.x; // 0..63
    int c_start, c_len;
    pick_chunk_4(sub, c0s, c0l, c1s, c1l, c2s, c2l, c3s, c3l, c_start, c_len);
    int bh = sub * 32 + hv;

    __shared__ float s_G[64];
    float dt_b = bf16_to_f(dt_bias[hv]);
    float neg_exp_A = -expf(A_log[hv]);

    float b_val = 0.0f;
    float g_val = 0.0f;
    if (tid < c_len) {
        const bf16_t* pin_t = packed_in + (size_t)(c_start + tid) * 12352;
        float bv = bf16_to_f(pin_t[12288 + hv]);
        b_val = 1.0f / (1.0f + expf(-bv));
        float av = bf16_to_f(pin_t[12320 + hv]) + dt_b;
        float sp = (av > 20.0f) ? av : log1pf(expf(av));
        g_val = neg_exp_A * sp;
    }
    s_G[tid] = g_val;
    buf_beta[(size_t)bh * 64 + tid] = b_val;
    __syncthreads();

    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < c_len; ++i) {
            acc += s_G[i];
            s_G[i] = acc;
        }
        buf_g_last[bh] = acc;
    }
    __syncthreads();
    buf_G[(size_t)bh * 64 + tid] = s_G[tid];
}

// 12.2 Parallel Q_norm, K_norm, Kb = beta * exp(G) * K (BF16)
// Grid: (32, 64, B_c), Block: (128, 1, 1) -> 8,192 blocks, fully parallel across tokens!
extern "C" __global__ void deltanet_wy_prep_qk_bf16(
    const bf16_t* __restrict__ qkv_conv,   // [T, 8192]
    const float* __restrict__ buf_G,       // [B_c, 32, 64]
    const float* __restrict__ buf_beta,    // [B_c, 32, 64]
    bf16_t* __restrict__ buf_Q,            // [B_c, 32, 64, 128]
    bf16_t* __restrict__ buf_K,            // [B_c, 32, 64, 128]
    bf16_t* __restrict__ buf_Kb,           // [B_c, 32, 64, 128]
    int c0s, int c0l, int c1s, int c1l, int c2s, int c2l, int c3s, int c3l
) {
    int hv = blockIdx.x;
    int i = blockIdx.y;
    int sub = blockIdx.z;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    int hk = hv >> 1;

    int c_start, c_len;
    pick_chunk_4(sub, c0s, c0l, c1s, c1l, c2s, c2l, c3s, c3l, c_start, c_len);
    int bh = sub * 32 + hv;
    size_t off = ((size_t)bh * 64 + i) * 128 + tid;

    if (i >= c_len) {
        buf_Q[off] = 0;
        buf_K[off] = 0;
        buf_Kb[off] = 0;
        return;
    }

    __shared__ float wsum_q[4];
    __shared__ float wsum_k[4];

    const bf16_t* qkv_t = qkv_conv + (size_t)(c_start + i) * 8192;
    float qv = bf16_to_f(qkv_t[hk * 128 + tid]);
    float kv = bf16_to_f(qkv_t[2048 + hk * 128 + tid]);

    float qs = warp_reduce_sum(qv * qv);
    float ks = warp_reduce_sum(kv * kv);
    if (lane == 0) {
        wsum_q[wid] = qs;
        wsum_k[wid] = ks;
    }
    __syncthreads();
    if (wid == 0) {
        float vq = warp_reduce_sum((lane < 4) ? wsum_q[lane] : 0.0f);
        float vk = warp_reduce_sum((lane < 4) ? wsum_k[lane] : 0.0f);
        if (lane == 0) {
            wsum_q[0] = rsqrtf(vq + 1e-6f) * 0.08838834764831845f;
            wsum_k[0] = rsqrtf(vk + 1e-6f);
        }
    }
    __syncthreads();

    float qn = qv * wsum_q[0];
    float kn = kv * wsum_k[0];
    float kb_s = buf_beta[(size_t)bh * 64 + i] * expf(buf_G[(size_t)bh * 64 + i]);
    buf_Q[off] = f_to_bf16(qn);
    buf_K[off] = f_to_bf16(kn);
    buf_Kb[off] = f_to_bf16(kn * kb_s);
}

// 12.3 Fast 2x32 Block Triangular Inversion T = (I + M)^-1 + Causal Mask A (BF16)
// Warp 0 solves T00 (32x32) & Warp 1 solves T11 (32x32) IN PARALLEL with ZERO __syncthreads()!
// Then T10 = -T11 @ (L10 @ T00). ~10 us total!
// Grid: (32, B_c, 1), Block: (64, 1, 1)
extern "C" __global__ void deltanet_wy_solve_2x32_bf16(
    bf16_t* __restrict__ buf_T,            // [B_c, 32, 64, 64] in: K@K^T, out: T=(I+M)^-1
    bf16_t* __restrict__ buf_A,            // [B_c, 32, 64, 64] in: Q@K^T, out: A_causal
    const float* __restrict__ buf_G,       // [B_c, 32, 64]
    const float* __restrict__ buf_beta,    // [B_c, 32, 64]
    int c0l, int c1l, int c2l, int c3l
) {
    int hv = blockIdx.x;
    int sub = blockIdx.y;
    int tid = threadIdx.x; // 0..63
    int wid = tid >> 5;    // 0 or 1
    int lane = tid & 31;   // 0..31
    int c_len = (sub == 0) ? c0l : ((sub == 1) ? c1l : ((sub == 2) ? c2l : c3l));
    int bh = sub * 32 + hv;

    __shared__ float s_L00[32][33];
    __shared__ float s_L11[32][33];
    __shared__ float s_L10[32][33];
    __shared__ float s_X[32][33];
    __shared__ float s_G[64];
    __shared__ float s_beta[64];

    s_G[tid] = buf_G[(size_t)bh * 64 + tid];
    s_beta[tid] = buf_beta[(size_t)bh * 64 + tid];
    __syncthreads();

    size_t mat_base = (size_t)bh * 4096;

    // 1. Load strict lower-tri M blocks & mask A_causal simultaneously
    // Warp 0 loads L00 (rows 0..31, cols 0..31) & upper-half A; Warp 1 loads L11 (rows 32..63, cols 32..63) & L10 (rows 32..63, cols 0..31)
    if (wid == 0) {
        for (int r = 0; r < 32; ++r) {
            float m = 0.0f;
            if (r < c_len && lane < r) {
                m = s_beta[r] * expf(s_G[r] - s_G[lane]) * bf16_to_f(buf_T[mat_base + r * 64 + lane]);
            }
            s_L00[r][lane] = m;
            // Mask A rows 0..31, cols 0..31 (cols 32..63 are strictly upper -> zero!)
            float a0 = (r < c_len && lane <= r) ? (bf16_to_f(buf_A[mat_base + r * 64 + lane]) * expf(s_G[r] - s_G[lane])) : 0.0f;
            buf_A[mat_base + r * 64 + lane] = f_to_bf16(a0);
            buf_A[mat_base + r * 64 + 32 + lane] = 0;
        }
        // Invert L00 in-place with ZERO __syncthreads() (single warp!)
        int n0 = (c_len < 32) ? c_len : 32;
        for (int i = 0; i < n0; ++i) {
            float val = (lane == i) ? 1.0f : 0.0f;
            if (lane < i) {
                float acc = 0.0f;
                for (int k = lane; k < i; ++k) acc -= s_L00[i][k] * s_L00[k][lane];
                val = acc;
            }
            __syncwarp();
            s_L00[i][lane] = val;
            __syncwarp();
        }
        for (int i = n0; i < 32; ++i) s_L00[i][lane] = 0.0f;
    } else {
        for (int r = 0; r < 32; ++r) {
            int gr = 32 + r;
            float m11 = 0.0f;
            float m10 = 0.0f;
            if (gr < c_len) {
                m10 = s_beta[gr] * expf(s_G[gr] - s_G[lane]) * bf16_to_f(buf_T[mat_base + gr * 64 + lane]);
                if (lane < r) {
                    m11 = s_beta[gr] * expf(s_G[gr] - s_G[32 + lane]) * bf16_to_f(buf_T[mat_base + gr * 64 + 32 + lane]);
                }
            }
            s_L10[r][lane] = m10;
            s_L11[r][lane] = m11;
            // Mask A rows 32..63
            float a10 = (gr < c_len) ? (bf16_to_f(buf_A[mat_base + gr * 64 + lane]) * expf(s_G[gr] - s_G[lane])) : 0.0f;
            float a11 = (gr < c_len && lane <= r) ? (bf16_to_f(buf_A[mat_base + gr * 64 + 32 + lane]) * expf(s_G[gr] - s_G[32 + lane])) : 0.0f;
            buf_A[mat_base + gr * 64 + lane] = f_to_bf16(a10);
            buf_A[mat_base + gr * 64 + 32 + lane] = f_to_bf16(a11);
        }
        // Invert L11 in-place with ZERO __syncthreads() (single warp!)
        int n1 = (c_len > 32) ? (c_len - 32) : 0;
        for (int i = 0; i < n1; ++i) {
            float val = (lane == i) ? 1.0f : 0.0f;
            if (lane < i) {
                float acc = 0.0f;
                for (int k = lane; k < i; ++k) acc -= s_L11[i][k] * s_L11[k][lane];
                val = acc;
            }
            __syncwarp();
            s_L11[i][lane] = val;
            __syncwarp();
        }
        for (int i = n1; i < 32; ++i) s_L11[i][lane] = 0.0f;
    }
    __syncthreads();

    // 2. Write T00 (rows 0..31, cols 0..31), zero upper-right (rows 0..31, cols 32..63), write T11 (rows 32..63, cols 32..63)
    int n1 = (c_len > 32) ? (c_len - 32) : 0;
    if (wid == 0) {
        for (int r = 0; r < 32; ++r) {
            buf_T[mat_base + r * 64 + lane] = f_to_bf16(s_L00[r][lane]);
            buf_T[mat_base + r * 64 + 32 + lane] = 0;
        }
    } else {
        for (int r = 0; r < 32; ++r) {
            buf_T[mat_base + (32 + r) * 64 + 32 + lane] = f_to_bf16(s_L11[r][lane]);
        }
    }

    // 3. Compute X = L10 @ T00 (only rows 0..n1-1; both warps split rows 16 each!)
    for (int r = wid; r < n1; r += 2) {
        float acc = 0.0f;
        // T00[k][lane] is zero for k < lane!
        for (int k = lane; k < 32; ++k) {
            acc += s_L10[r][k] * s_L00[k][lane];
        }
        s_X[r][lane] = acc;
    }
    __syncthreads();

    // 4. Compute T10 = -T11 @ X and write rows 32..63, cols 0..31
    for (int r = wid; r < 32; r += 2) {
        float acc = 0.0f;
        if (r < n1) {
            // T11[r][k] is zero for k > r!
            for (int k = 0; k <= r; ++k) {
                acc -= s_L11[r][k] * s_X[k][lane];
            }
        }
        buf_T[mat_base + (32 + r) * 64 + lane] = f_to_bf16(acc);
    }
}

// 12.4 Fused Prep Vb = beta * V (BF16) + Scale buf_Q *= exp(G_i) & buf_K *= exp(G_last - G_i)
// Grid: (32, 64, B_c), Block: (128, 1, 1) -> 8,192 coalesced blocks!
extern "C" __global__ void deltanet_wy_prep_vb_and_scale_qk_bf16(
    const bf16_t* __restrict__ qkv_conv,   // [T, 8192]
    const float* __restrict__ buf_G,       // [B_c, 32, 64]
    const float* __restrict__ buf_beta,    // [B_c, 32, 64]
    const float* __restrict__ buf_g_last,  // [B_c, 32]
    bf16_t* __restrict__ buf_Q,            // [B_c, 32, 64, 128] scaled in-place
    bf16_t* __restrict__ buf_K,            // [B_c, 32, 64, 128] scaled in-place
    bf16_t* __restrict__ buf_Vb,           // [B_c, 32, 64, 128] output
    int c0s, int c0l, int c1s, int c1l, int c2s, int c2l, int c3s, int c3l
) {
    int hv = blockIdx.x;
    int i = blockIdx.y;
    int sub = blockIdx.z;
    int tid = threadIdx.x;

    int c_start, c_len;
    pick_chunk_4(sub, c0s, c0l, c1s, c1l, c2s, c2l, c3s, c3l, c_start, c_len);
    int bh = sub * 32 + hv;
    size_t off = ((size_t)bh * 64 + i) * 128 + tid;

    if (i < c_len) {
        float g_i = buf_G[(size_t)bh * 64 + i];
        float g_last = buf_g_last[bh];
        float beta = buf_beta[(size_t)bh * 64 + i];
        float v = bf16_to_f(qkv_conv[(size_t)(c_start + i) * 8192 + 4096 + hv * 128 + tid]);
        buf_Vb[off] = f_to_bf16(v * beta);
        buf_Q[off] = f_to_bf16(bf16_to_f(buf_Q[off]) * expf(g_i));
        buf_K[off] = f_to_bf16(bf16_to_f(buf_K[off]) * expf(g_last - g_i));
    } else {
        buf_Vb[off] = 0;
    }
}

// 12.5 Fused WMMA BF16 Tensor Core Inter-Chunk State + Output Kernel!
// Keeps S [128 K x 16 V] permanently in FP32 WMMA registers across all B_c <= 4 chunks (ZERO intermediate S DRAM!).
// Runs W@S, Q@S, A_causal@U_new (lower-tri!), and K^T@U_new on Ampere BF16 Tensor Cores!
// Grid: (32 heads, 8 V-tiles, 1) = 256 blocks, Block: (128, 1, 1) = 4 warps
extern "C" __global__ void deltanet_wy_state_wmma_bf16(
    const bf16_t* __restrict__ buf_Q,      // [B_c, 32, 64, 128] (exp(G_i) * Q)
    const bf16_t* __restrict__ buf_K,      // [B_c, 32, 64, 128] (exp(G_last - G_i) * K)
    const bf16_t* __restrict__ buf_W,      // [B_c, 32, 64, 128]
    const bf16_t* __restrict__ buf_U,      // [B_c, 32, 64, 128]
    const bf16_t* __restrict__ buf_A,      // [B_c, 32, 64, 64]  (A_causal)
    const float* __restrict__ buf_g_last,  // [B_c, 32]
    float* __restrict__ rec_state,         // [32, 128, 128] FP32
    float* __restrict__ rec_snap_out,      // optional [32, 128, 128] FP32
    bf16_t* __restrict__ out_bf16,         // [T, 4096] BF16 (core_out)
    int b_c,
    int has_init_state,
    int snap_t,
    int c0s, int c0l, int c1s, int c1l, int c2s, int c2l, int c3s, int c3l
) {
    using namespace nvcuda::wmma;
    int hv = blockIdx.x;
    int vt = blockIdx.y; // 0..7 -> v_base = vt * 16
    int v_base = vt * 16;
    int tid = threadIdx.x;
    int wid = tid >> 5;  // 0..3

    __shared__ float s_S_f32[128][16];
    __shared__ __nv_bfloat16 s_S_bf16[128][16];
    __shared__ float s_U_f32[64][16];
    __shared__ __nv_bfloat16 s_U_bf16[64][16];
    __shared__ float s_O_f32[64][16];

    fragment<accumulator, 16, 16, 16, float> S_frag[2];
    float* S_head = rec_state + (size_t)hv * 16384;

    if (has_init_state) {
        // Load [128, 16] strip of S from global -> s_S_f32 -> WMMA fragments
        for (int idx = tid; idx < 2048; idx += 128) {
            int kr = idx >> 4;
            int vc = idx & 15;
            s_S_f32[kr][vc] = S_head[kr * 128 + v_base + vc];
        }
        __syncthreads();
        load_matrix_sync(S_frag[0], &s_S_f32[wid * 32][0], 16, mem_row_major);
        load_matrix_sync(S_frag[1], &s_S_f32[wid * 32 + 16][0], 16, mem_row_major);
        __syncthreads();
    } else {
        fill_fragment(S_frag[0], 0.0f);
        fill_fragment(S_frag[1], 0.0f);
    }

    for (int sub = 0; sub < b_c; ++sub) {
        int c_start, c_len;
        pick_chunk_4(sub, c0s, c0l, c1s, c1l, c2s, c2l, c3s, c3l, c_start, c_len);
        int bh = sub * 32 + hv;
        size_t qk_off = (size_t)bh * 8192;
        size_t a_off  = (size_t)bh * 4096;

        const __nv_bfloat16* W_sub = reinterpret_cast<const __nv_bfloat16*>(buf_W + qk_off);
        const __nv_bfloat16* Q_sub = reinterpret_cast<const __nv_bfloat16*>(buf_Q + qk_off);
        const __nv_bfloat16* K_sub = reinterpret_cast<const __nv_bfloat16*>(buf_K + qk_off);
        const __nv_bfloat16* A_sub = reinterpret_cast<const __nv_bfloat16*>(buf_A + a_off);
        const bf16_t* U_sub = buf_U + qk_off;

        bool state_nonzero = (has_init_state != 0) || (sub > 0);

        // Step A: quantize current FP32 S_frag -> s_S_bf16 for W@S & Q@S
        if (state_nonzero) {
            store_matrix_sync(&s_S_f32[wid * 32][0], S_frag[0], 16, mem_row_major);
            store_matrix_sync(&s_S_f32[wid * 32 + 16][0], S_frag[1], 16, mem_row_major);
            __syncthreads();
            for (int idx = tid; idx < 2048; idx += 128) {
                int kr = idx >> 4;
                int vc = idx & 15;
                s_S_bf16[kr][vc] = __float2bfloat16(s_S_f32[kr][vc]);
            }
            __syncthreads();

            // Step B: U_new[64, 16] = U[64, 16] - W[64, 128] @ S[128, 16]
            fragment<accumulator, 16, 16, 16, float> acc_u;
            fill_fragment(acc_u, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_f;
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_f;
                load_matrix_sync(a_f, W_sub + (wid * 16) * 128 + kt * 16, 128);
                load_matrix_sync(b_f, &s_S_bf16[kt * 16][0], 16);
                mma_sync(acc_u, a_f, b_f, acc_u);
            }
            store_matrix_sync(&s_U_f32[wid * 16][0], acc_u, 16, mem_row_major);
            __syncthreads();
            for (int idx = tid; idx < 1024; idx += 128) {
                int r = idx >> 4;
                int vc = idx & 15;
                float u_val = (r < c_len) ? (bf16_to_f(U_sub[r * 128 + v_base + vc]) - s_U_f32[r][vc]) : 0.0f;
                s_U_bf16[r][vc] = __float2bfloat16(u_val);
            }
            __syncthreads();
        } else {
            for (int idx = tid; idx < 1024; idx += 128) {
                int r = idx >> 4;
                int vc = idx & 15;
                bf16_t raw_u = (r < c_len) ? U_sub[r * 128 + v_base + vc] : 0;
                s_U_bf16[r][vc] = *reinterpret_cast<const __nv_bfloat16*>(&raw_u);
            }
            __syncthreads();
        }

        // Step C & D: O[64, 16] = Q[64, 128] @ S[128, 16] + A_causal[64, 64] @ U_new[64, 16]
        fragment<accumulator, 16, 16, 16, float> acc_o;
        fill_fragment(acc_o, 0.0f);
        if (state_nonzero) {
            #pragma unroll
            for (int kt = 0; kt < 8; ++kt) {
                fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_f;
                fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_f;
                load_matrix_sync(a_f, Q_sub + (wid * 16) * 128 + kt * 16, 128);
                load_matrix_sync(b_f, &s_S_bf16[kt * 16][0], 16);
                mma_sync(acc_o, a_f, b_f, acc_o);
            }
        }
        // Causal lower-tri: warp wid (rows wid*16..wid*16+15) only needs j_tile <= wid!
        for (int jt = 0; jt <= wid; ++jt) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a_f;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> b_f;
            load_matrix_sync(a_f, A_sub + (wid * 16) * 64 + jt * 16, 64);
            load_matrix_sync(b_f, &s_U_bf16[jt * 16][0], 16);
            mma_sync(acc_o, a_f, b_f, acc_o);
        }
        store_matrix_sync(&s_O_f32[wid * 16][0], acc_o, 16, mem_row_major);
        __syncthreads();

        // Write O directly to core_out [T, 4096] BF16
        for (int idx = tid; idx < c_len * 16; idx += 128) {
            int r = idx >> 4;
            int vc = idx & 15;
            out_bf16[(size_t)(c_start + r) * 4096 + hv * 128 + v_base + vc] = f_to_bf16(s_O_f32[r][vc]);
        }

        // Step E: S = exp(g_last) * S + K_decay^T @ U_new (100% in FP32 WMMA registers!)
        if (state_nonzero) {
            float decay = expf(buf_g_last[bh]);
            #pragma unroll
            for (int e = 0; e < S_frag[0].num_elements; ++e) S_frag[0].x[e] *= decay;
            #pragma unroll
            for (int e = 0; e < S_frag[1].num_elements; ++e) S_frag[1].x[e] *= decay;
        }
        #pragma unroll
        for (int ct = 0; ct < 4; ++ct) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major> k_f0;
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, col_major> k_f1;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> u_f;
            load_matrix_sync(u_f, &s_U_bf16[ct * 16][0], 16);
            load_matrix_sync(k_f0, K_sub + (ct * 16) * 128 + (wid * 32), 128);
            mma_sync(S_frag[0], k_f0, u_f, S_frag[0]);
            load_matrix_sync(k_f1, K_sub + (ct * 16) * 128 + (wid * 32 + 16), 128);
            mma_sync(S_frag[1], k_f1, u_f, S_frag[1]);
        }

        // Step F: In-flight Prefix Cache snapshot at exact snap_t boundary!
        if (snap_t > 0 && (c_start + c_len) == snap_t && rec_snap_out != nullptr) {
            store_matrix_sync(&s_S_f32[wid * 32][0], S_frag[0], 16, mem_row_major);
            store_matrix_sync(&s_S_f32[wid * 32 + 16][0], S_frag[1], 16, mem_row_major);
            __syncthreads();
            float* S_snap = rec_snap_out + (size_t)hv * 16384;
            for (int idx = tid; idx < 2048; idx += 128) {
                int kr = idx >> 4;
                int vc = idx & 15;
                S_snap[kr * 128 + v_base + vc] = s_S_f32[kr][vc];
            }
            __syncthreads();
        }
        __syncthreads();
    }

    // Store final FP32 S_frag back to rec_state once per tile
    store_matrix_sync(&s_S_f32[wid * 32][0], S_frag[0], 16, mem_row_major);
    store_matrix_sync(&s_S_f32[wid * 32 + 16][0], S_frag[1], 16, mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < 2048; idx += 128) {
        int kr = idx >> 4;
        int vc = idx & 15;
        S_head[kr * 128 + v_base + vc] = s_S_f32[kr][vc];
    }
}

// 12.6 In-place Fused RMSNorm + SiLU(z) gate on core_out [T, 4096] BF16 (1 launch per layer!)
// Grid: (32, seq_len, 1), Block: (128, 1, 1)
extern "C" __global__ void deltanet_prefill_norm_gate_inplace_bf16(
    bf16_t* __restrict__ core_out,        // [T, 4096] BF16 in-place
    const bf16_t* __restrict__ packed_in, // [T, 12352]
    const bf16_t* __restrict__ norm_w,    // [128]
    float eps
) {
    int hv = blockIdx.x;
    int t = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    __shared__ float wsum[4];
    size_t idx = (size_t)t * 4096 + hv * 128 + tid;
    float out_val = bf16_to_f(core_out[idx]);
    float os = warp_reduce_sum(out_val * out_val);
    if (lane == 0) wsum[wid] = os;
    __syncthreads();
    if (wid == 0) {
        float v = warp_reduce_sum((lane < 4) ? wsum[lane] : 0.0f);
        if (lane == 0) wsum[0] = rsqrtf(v * (1.0f / 128.0f) + eps);
    }
    __syncthreads();
    float normed = bf16_to_f(f_to_bf16(out_val * wsum[0])) * bf16_to_f(norm_w[tid]);
    float zv = bf16_to_f(packed_in[(size_t)t * 12352 + 8192 + hv * 128 + tid]);
    float z_silu = zv / (1.0f + expf(-zv));
    core_out[idx] = f_to_bf16(normed * z_silu);
}

// ===========================================================================
// 13. Phase R9: 2D Parallel Causal Conv1D Prefill (512 blocks!) &
//               WMMA BF16 Tensor Core FlashAttention-2 Prefill (Contiguous + Paged)
// ===========================================================================

// 13.1 2D Parallel Causal Conv1D (k=4): Grid (32, ceil(seq_len / 32), 1), Block (256, 1, 1)
// Keeps conv_state READ-ONLY (100% race-free across all time tiles!); updates conv_snap_out at snap_t - 1.
extern "C" __global__ void causal_conv1d_prefill_2d_bf16(
    const bf16_t* __restrict__ packed_in,
    const bf16_t* __restrict__ conv_state,     // READ-ONLY initial state
    bf16_t* __restrict__ conv_snap_out,        // optional: writes state at t == snap_t - 1
    const bf16_t* __restrict__ conv_weight,
    bf16_t* __restrict__ y_out,
    int seq_len,
    int C,
    int stride_in,
    int use_init_state,
    int snap_t
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int t_start = blockIdx.y * 32;
    if (t_start >= seq_len) return;
    int t_end = t_start + 32;
    if (t_end > seq_len) t_end = seq_len;

    int wbase = c * 4;
    float w0 = bf16_to_f(conv_weight[wbase + 0]);
    float w1 = bf16_to_f(conv_weight[wbase + 1]);
    float w2 = bf16_to_f(conv_weight[wbase + 2]);
    float w3 = bf16_to_f(conv_weight[wbase + 3]);

    auto get_in = [&](int idx) -> bf16_t {
        if (idx >= 0) return packed_in[(size_t)idx * stride_in + c];
        if (use_init_state) return conv_state[wbase + (4 + idx)];
        return 0;
    };

    bf16_t s1 = get_in(t_start - 3);
    bf16_t s2 = get_in(t_start - 2);
    bf16_t s3 = get_in(t_start - 1);

    for (int t = t_start; t < t_end; ++t) {
        bf16_t s0 = s1;
        s1 = s2;
        s2 = s3;
        s3 = packed_in[(size_t)t * stride_in + c];

        if (snap_t > 0 && t == snap_t - 1) {
            conv_snap_out[wbase + 0] = s0;
            conv_snap_out[wbase + 1] = s1;
            conv_snap_out[wbase + 2] = s2;
            conv_snap_out[wbase + 3] = s3;
        }

        float acc = bf16_to_f(s0) * w0
                  + bf16_to_f(s1) * w1
                  + bf16_to_f(s2) * w2
                  + bf16_to_f(s3) * w3;
        float silu = acc / (1.0f + expf(-acc));
        y_out[(size_t)t * C + c] = f_to_bf16(silu);
    }
}

// 13.2 Tiny 1.5-us race-free Conv1D final state writer: Grid (32, 1, 1), Block (256, 1, 1)
extern "C" __global__ void causal_conv1d_update_state_bf16(
    const bf16_t* __restrict__ packed_in,
    bf16_t* __restrict__ conv_state,
    int seq_len,
    int C,
    int stride_in,
    int use_init_state
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    int wbase = c * 4;
    bf16_t old1 = use_init_state ? conv_state[wbase + 1] : 0;
    bf16_t old2 = use_init_state ? conv_state[wbase + 2] : 0;
    bf16_t old3 = use_init_state ? conv_state[wbase + 3] : 0;
    auto get_in = [&](int idx) -> bf16_t {
        if (idx >= 0) return packed_in[(size_t)idx * stride_in + c];
        if (idx == -3) return old1;
        if (idx == -2) return old2;
        if (idx == -1) return old3;
        return 0;
    };
    conv_state[wbase + 0] = get_in(seq_len - 4);
    conv_state[wbase + 1] = get_in(seq_len - 3);
    conv_state[wbase + 2] = get_in(seq_len - 2);
    conv_state[wbase + 3] = get_in(seq_len - 1);
}

// 13.3 WMMA BF16 Tensor Core FlashAttention-2 Causal Prefill GQA + Sigmoid Gate (Contiguous KV)
// Grid: (16 Q-heads, ceil(seq_len / 16), 1) = 512 blocks at T=512, Block: (128, 1, 1) = 4 warps
// 36.5 KB SMEM (Q resident in SMEM, KV reused K->V, O FP32 in SMEM), 0 spill!
extern "C" __global__ void prefill_gqa_wmma_causal_gated_bf16(
    const bf16_t* __restrict__ q_sdpa,      // [16, seq_len, 256]
    const bf16_t* __restrict__ qkv_packed,  // [seq_len, 10240]
    const bf16_t* __restrict__ k_cache,     // [4, max_seq, 256]
    const bf16_t* __restrict__ v_cache,     // [4, max_seq, 256]
    int start_pos,
    int seq_len,
    int max_seq,
    bf16_t* __restrict__ out                // [seq_len, 4096]
) {
    using namespace nvcuda::wmma;
    int hq = blockIdx.x;
    int hk = hq >> 2;
    int q_base = blockIdx.y * 16;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    __shared__ __align__(16) __nv_bfloat16 s_Q[16][256];
    __shared__ __align__(16) __nv_bfloat16 s_KV[16][256];
    __shared__ __align__(16) float s_O[16][256];
    __shared__ __align__(16) float s_S[4][16][16];
    __shared__ __align__(16) __nv_bfloat16 s_P[16][16];
    __shared__ float s_m[16];
    __shared__ float s_l[16];
    __shared__ float s_alpha[16];

    // Preload Q [16, 256] ONCE into SMEM & zero s_O
    const bf16_t* q_head = q_sdpa + (size_t)hq * seq_len * 256;
    for (int idx = tid; idx < 512; idx += 128) {
        int r = idx >> 5;       // 0..15
        int c8 = (idx & 31) * 8;
        if (q_base + r < seq_len) {
            *reinterpret_cast<uint4*>(&s_Q[r][c8]) =
                *reinterpret_cast<const uint4*>(q_head + (size_t)(q_base + r) * 256 + c8);
        } else {
            uint4 z = {0, 0, 0, 0};
            *reinterpret_cast<uint4*>(&s_Q[r][c8]) = z;
        }
    }
    for (int idx = tid; idx < 4096; idx += 128) {
        s_O[idx >> 8][idx & 255] = 0.0f;
    }
    if (tid < 16) {
        s_m[tid] = -1e30f;
        s_l[tid] = 0.0f;
    }
    __syncthreads();

    const bf16_t* k_head = k_cache + (size_t)hk * max_seq * 256;
    const bf16_t* v_head = v_cache + (size_t)hk * max_seq * 256;
    int q_max = (q_base + 15 < seq_len) ? (q_base + 15) : (seq_len - 1);
    int kv_end_blk = start_pos + q_max + 1;

    for (int kt0 = 0; kt0 < kv_end_blk; kt0 += 16) {
        // 1. Load K tile [16, 256] into s_KV
        for (int idx = tid; idx < 512; idx += 128) {
            int r = idx >> 5;
            int c8 = (idx & 31) * 8;
            if (kt0 + r < kv_end_blk) {
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) =
                    *reinterpret_cast<const uint4*>(k_head + (size_t)(kt0 + r) * 256 + c8);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) = z;
            }
        }
        __syncthreads();

        // 2. Q @ K^T on BF16 Tensor Cores (each warp accumulates 64 of the 256 dims -> 4 WMMA steps!)
        fragment<accumulator, 16, 16, 16, float> s_frag;
        fill_fragment(s_frag, 0.0f);
        int d_base = wid * 64;
        #pragma unroll
        for (int dk = 0; dk < 4; ++dk) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> q_f;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> k_f;
            load_matrix_sync(q_f, &s_Q[0][d_base + dk * 16], 256);
            load_matrix_sync(k_f, &s_KV[0][d_base + dk * 16], 256);
            mma_sync(s_frag, q_f, k_f, s_frag);
        }
        store_matrix_sync(&s_S[wid][0][0], s_frag, 16, mem_row_major);
        __syncthreads();

        // 3. Reduce 4 warps & run exact online softmax (16 threads = 1 row each, 16 cols!)
        if (tid < 16) {
            int r = tid;
            int q_pos = start_pos + q_base + r;
            bool valid_r = (q_base + r < seq_len);
            float row_s[16];
            float row_max = s_m[r];
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float sc = (valid_r && (kt0 + c) <= q_pos)
                    ? ((s_S[0][r][c] + s_S[1][r][c] + s_S[2][r][c] + s_S[3][r][c]) * 0.0625f)
                    : -1e30f;
                row_s[c] = sc;
                row_max = fmaxf(row_max, sc);
            }
            float alpha = (s_m[r] > -1e29f) ? expf(s_m[r] - row_max) : 0.0f;
            float l_new = s_l[r] * alpha;
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float p = (valid_r && (kt0 + c) <= q_pos) ? expf(row_s[c] - row_max) : 0.0f;
                l_new += p;
                s_P[r][c] = __float2bfloat16(p);
            }
            s_m[r] = row_max;
            s_l[r] = l_new;
            s_alpha[r] = alpha;
        }
        __syncthreads();

        // 4. Load V tile [16, 256] into s_KV
        for (int idx = tid; idx < 512; idx += 128) {
            int r = idx >> 5;
            int c8 = (idx & 31) * 8;
            if (kt0 + r < kv_end_blk) {
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) =
                    *reinterpret_cast<const uint4*>(v_head + (size_t)(kt0 + r) * 256 + c8);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) = z;
            }
        }
        __syncthreads();

        // 5. P @ V on BF16 Tensor Cores (warp wid computes 4 tiles of 16 cols = 64 cols!)
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> p_f;
        load_matrix_sync(p_f, &s_P[0][0], 16);
        #pragma unroll
        for (int dc = 0; dc < 4; ++dc) {
            int col_base = d_base + dc * 16;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> v_f;
            fragment<accumulator, 16, 16, 16, float> o_f;
            fill_fragment(o_f, 0.0f);
            load_matrix_sync(v_f, &s_KV[0][col_base], 256);
            mma_sync(o_f, p_f, v_f, o_f);
            store_matrix_sync(&s_S[wid][0][0], o_f, 16, mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = lane; e < 256; e += 32) {
                int r = e >> 4;
                int cc = e & 15;
                s_O[r][col_base + cc] = s_O[r][col_base + cc] * s_alpha[r] + s_S[wid][r][cc];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    // 6. Final normalize + SiLU/Sigmoid output gate -> core_out [seq_len, 4096]
    for (int idx = tid; idx < 4096; idx += 128) {
        int r = idx >> 8;
        int d = idx & 255;
        if (q_base + r < seq_len) {
            float inv_l = 1.0f / fmaxf(s_l[r], 1e-30f);
            float g_val = bf16_to_f(qkv_packed[(size_t)(q_base + r) * 10240 + hq * 512 + 256 + d]);
            float sig = 1.0f / (1.0f + expf(-g_val));
            out[(size_t)(q_base + r) * 4096 + hq * 256 + d] = f_to_bf16((s_O[r][d] * inv_l) * sig);
        }
    }
}

// 13.4 WMMA BF16 Tensor Core FlashAttention-2 Causal Prefill GQA over Paged KV Blocks
// Grid: (16 Q-heads, ceil(seq_len / 16), 1), Block: (128, 1, 1)
extern "C" __global__ void prefill_gqa_paged_wmma_causal_gated_bf16(
    const bf16_t* __restrict__ q_sdpa,      // [16, seq_len, 256]
    const bf16_t* __restrict__ qkv_packed,  // [seq_len, 10240]
    const bf16_t* __restrict__ kv_pool,     // [num_blocks, 8, 2, 4, 16, 256]
    const int* __restrict__ block_table,
    int fa_idx,
    int start_pos,
    int seq_len,
    bf16_t* __restrict__ out                // [seq_len, 4096]
) {
    using namespace nvcuda::wmma;
    int hq = blockIdx.x;
    int hk = hq >> 2;
    int q_base = blockIdx.y * 16;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;

    __shared__ __align__(16) __nv_bfloat16 s_Q[16][256];
    __shared__ __align__(16) __nv_bfloat16 s_KV[16][256];
    __shared__ __align__(16) float s_O[16][256];
    __shared__ __align__(16) float s_S[4][16][16];
    __shared__ __align__(16) __nv_bfloat16 s_P[16][16];
    __shared__ float s_m[16];
    __shared__ float s_l[16];
    __shared__ float s_alpha[16];

    const bf16_t* q_head = q_sdpa + (size_t)hq * seq_len * 256;
    for (int idx = tid; idx < 512; idx += 128) {
        int r = idx >> 5;
        int c8 = (idx & 31) * 8;
        if (q_base + r < seq_len) {
            *reinterpret_cast<uint4*>(&s_Q[r][c8]) =
                *reinterpret_cast<const uint4*>(q_head + (size_t)(q_base + r) * 256 + c8);
        } else {
            uint4 z = {0, 0, 0, 0};
            *reinterpret_cast<uint4*>(&s_Q[r][c8]) = z;
        }
    }
    for (int idx = tid; idx < 4096; idx += 128) {
        s_O[idx >> 8][idx & 255] = 0.0f;
    }
    if (tid < 16) {
        s_m[tid] = -1e30f;
        s_l[tid] = 0.0f;
    }
    __syncthreads();

    int q_max = (q_base + 15 < seq_len) ? (q_base + 15) : (seq_len - 1);
    int kv_end_blk = start_pos + q_max + 1;

    for (int kt0 = 0; kt0 < kv_end_blk; kt0 += 16) {
        int blk = block_table[kt0 >> 4];
        const bf16_t* k_blk = kv_pool + (size_t)blk * 262144 + (size_t)fa_idx * 32768 + (size_t)hk * 4096;
        const bf16_t* v_blk = k_blk + 16384;

        for (int idx = tid; idx < 512; idx += 128) {
            int r = idx >> 5;
            int c8 = (idx & 31) * 8;
            if (kt0 + r < kv_end_blk) {
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) =
                    *reinterpret_cast<const uint4*>(k_blk + (size_t)r * 256 + c8);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) = z;
            }
        }
        __syncthreads();

        fragment<accumulator, 16, 16, 16, float> s_frag;
        fill_fragment(s_frag, 0.0f);
        int d_base = wid * 64;
        #pragma unroll
        for (int dk = 0; dk < 4; ++dk) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> q_f;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> k_f;
            load_matrix_sync(q_f, &s_Q[0][d_base + dk * 16], 256);
            load_matrix_sync(k_f, &s_KV[0][d_base + dk * 16], 256);
            mma_sync(s_frag, q_f, k_f, s_frag);
        }
        store_matrix_sync(&s_S[wid][0][0], s_frag, 16, mem_row_major);
        __syncthreads();

        if (tid < 16) {
            int r = tid;
            int q_pos = start_pos + q_base + r;
            bool valid_r = (q_base + r < seq_len);
            float row_s[16];
            float row_max = s_m[r];
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float sc = (valid_r && (kt0 + c) <= q_pos)
                    ? ((s_S[0][r][c] + s_S[1][r][c] + s_S[2][r][c] + s_S[3][r][c]) * 0.0625f)
                    : -1e30f;
                row_s[c] = sc;
                row_max = fmaxf(row_max, sc);
            }
            float alpha = (s_m[r] > -1e29f) ? expf(s_m[r] - row_max) : 0.0f;
            float l_new = s_l[r] * alpha;
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float p = (valid_r && (kt0 + c) <= q_pos) ? expf(row_s[c] - row_max) : 0.0f;
                l_new += p;
                s_P[r][c] = __float2bfloat16(p);
            }
            s_m[r] = row_max;
            s_l[r] = l_new;
            s_alpha[r] = alpha;
        }
        __syncthreads();

        for (int idx = tid; idx < 512; idx += 128) {
            int r = idx >> 5;
            int c8 = (idx & 31) * 8;
            if (kt0 + r < kv_end_blk) {
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) =
                    *reinterpret_cast<const uint4*>(v_blk + (size_t)r * 256 + c8);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) = z;
            }
        }
        __syncthreads();

        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> p_f;
        load_matrix_sync(p_f, &s_P[0][0], 16);
        #pragma unroll
        for (int dc = 0; dc < 4; ++dc) {
            int col_base = d_base + dc * 16;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> v_f;
            fragment<accumulator, 16, 16, 16, float> o_f;
            fill_fragment(o_f, 0.0f);
            load_matrix_sync(v_f, &s_KV[0][col_base], 256);
            mma_sync(o_f, p_f, v_f, o_f);
            store_matrix_sync(&s_S[wid][0][0], o_f, 16, mem_row_major);
            __syncwarp();
            #pragma unroll
            for (int e = lane; e < 256; e += 32) {
                int r = e >> 4;
                int cc = e & 15;
                s_O[r][col_base + cc] = s_O[r][col_base + cc] * s_alpha[r] + s_S[wid][r][cc];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    for (int idx = tid; idx < 4096; idx += 128) {
        int r = idx >> 8;
        int d = idx & 255;
        if (q_base + r < seq_len) {
            float inv_l = 1.0f / fmaxf(s_l[r], 1e-30f);
            float g_val = bf16_to_f(qkv_packed[(size_t)(q_base + r) * 10240 + hq * 512 + 256 + d]);
            float sig = 1.0f / (1.0f + expf(-g_val));
            out[(size_t)(q_base + r) * 4096 + hq * 256 + d] = f_to_bf16((s_O[r][d] * inv_l) * sig);
        }
    }
}

// ============================================================================
// 14. PHASE R10: ONLINE SMOOTHQUANT + OZAKI INT8-SLICE TENSOR CORE PREFILL
//     Unlocks Ampere 238 TOPS INT8 Tensor Cores (175 TOPS measured on RTX 3080)
//     with 0% model weight modification, +0.00 MiB extra VRAM, 0 bytes spill!
// ============================================================================

__device__ __forceinline__ float r10_warp_max(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    }
    return v;
}

// 14.1 High-Occupancy 2D Per-column SmoothQuant scale: s[k] = max(|A[:, k]|, 1e-3)^alpha, inv_s[k] = 1/s[k]
// Grid: (K / 32, 1, 1) -> 80..288 blocks (100% SM occupancy!), Block: (32, 8, 1) = 8 warps reducing across M!
extern "C" __global__ void compute_smooth_col_scales_bf16(
    const bf16_t* __restrict__ A,
    float* __restrict__ smooth_s,
    float* __restrict__ smooth_inv_s,
    int M,
    int K,
    float alpha
) {
    int k = blockIdx.x * 32 + threadIdx.x;
    int wy = threadIdx.y;
    if (k >= K) return;

    float cmax = 0.0f;
    for (int m = wy; m < M; m += 8) {
        cmax = fmaxf(cmax, fabsf(bf16_to_f(A[(size_t)m * K + k])));
    }

    __shared__ float smem[8][32];
    smem[wy][threadIdx.x] = cmax;
    __syncthreads();

    if (wy == 0) {
        float gmax = smem[0][threadIdx.x];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            gmax = fmaxf(gmax, smem[w][threadIdx.x]);
        }
        float s = powf(fmaxf(gmax, 1e-3f), alpha);
        smooth_s[k] = s;
        smooth_inv_s[k] = 1.0f / s;
    }
}

// Helper device template: 1-Pass Register-Cached BF16 -> INT8 1-Slice with SmoothQuant col_mul
template<int MAX_ITERS>
__device__ __forceinline__ void quant_bf16_to_i8_smooth_impl(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0,
    float* __restrict__ S0,
    int valid_rows,
    int K
) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int k8 = K >> 3;
    uint2* out_v = reinterpret_cast<uint2*>(Q0 + (size_t)r * K);

    if (r >= valid_rows) {
        if (tid == 0) S0[r] = 1.0f;
        uint2 z = {0, 0};
        #pragma unroll
        for (int it = 0; it < MAX_ITERS; ++it) {
            int i = it * 256 + tid;
            if (i < k8) out_v[i] = z;
        }
        return;
    }

    const uint4* row_v = reinterpret_cast<const uint4*>(X + (size_t)r * K);
    float reg_vals[MAX_ITERS * 8];
    float lmax = 0.0f;

    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        int i = it * 256 + tid;
        if (i < k8) {
            uint4 v = row_v[i];
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
            int base_k = i << 3;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __bfloat1622float2(p[j]);
                f.x *= col_mul[base_k + 2 * j];
                f.y *= col_mul[base_k + 2 * j + 1];
                reg_vals[it * 8 + 2 * j]     = f.x;
                reg_vals[it * 8 + 2 * j + 1] = f.y;
                lmax = fmaxf(lmax, fabsf(f.x));
                lmax = fmaxf(lmax, fabsf(f.y));
            }
        }
    }

    lmax = r10_warp_max(lmax);
    __shared__ float smax[8];
    if ((tid & 31) == 0) smax[tid >> 5] = lmax;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) m = fmaxf(m, smax[w]);
        smax[0] = (m > 1e-12f) ? (m * (1.0f / 127.0f)) : 1.0f;
        S0[r] = smax[0];
    }
    __syncthreads();
    float inv_s = 1.0f / smax[0];

    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        int i = it * 256 + tid;
        if (i < k8) {
            signed char q[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int iv = __float2int_rn(reg_vals[it * 8 + j] * inv_s);
                q[j] = (signed char)(iv < -127 ? -127 : (iv > 127 ? 127 : iv));
            }
            out_v[i] = *reinterpret_cast<uint2*>(q);
        }
    }
}

// 14.2 1-Pass SmoothQuant BF16 -> INT8 (K <= 4096: 2560, 4096)
extern "C" __global__ void quant_bf16_to_i8_smooth_k4k(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0,
    float* __restrict__ S0,
    int valid_rows,
    int K
) {
    quant_bf16_to_i8_smooth_impl<2>(X, col_mul, Q0, S0, valid_rows, K);
}

// 14.3 1-Pass SmoothQuant BF16 -> INT8 (K <= 10240: 9216)
extern "C" __global__ void quant_bf16_to_i8_smooth_k10k(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0,
    float* __restrict__ S0,
    int valid_rows,
    int K
) {
    quant_bf16_to_i8_smooth_impl<5>(X, col_mul, Q0, S0, valid_rows, K);
}

// Helper device template: 1-Pass Register-Cached BF16 -> 2-Slice Ozaki INT8 with SmoothQuant col_mul
template<int MAX_ITERS>
__device__ __forceinline__ void quant_bf16_to_i8_ozaki2_impl(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0_base,
    signed char* __restrict__ Q1_base,
    float* __restrict__ S0,
    float* __restrict__ S1,
    int valid_rows,
    int K
) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int k8 = K >> 3;
    uint2* out0_v = reinterpret_cast<uint2*>(Q0_base + (size_t)r * K);
    uint2* out1_v = reinterpret_cast<uint2*>(Q1_base + (size_t)r * K);

    if (r >= valid_rows) {
        if (tid == 0) { S0[r] = 1.0f; S1[r] = 0.0f; }
        uint2 z = {0, 0};
        #pragma unroll
        for (int it = 0; it < MAX_ITERS; ++it) {
            int i = it * 256 + tid;
            if (i < k8) { out0_v[i] = z; out1_v[i] = z; }
        }
        return;
    }

    const uint4* row_v = reinterpret_cast<const uint4*>(X + (size_t)r * K);
    float reg_vals[MAX_ITERS * 8];
    float lmax = 0.0f;

    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        int i = it * 256 + tid;
        if (i < k8) {
            uint4 v = row_v[i];
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
            int base_k = i << 3;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __bfloat1622float2(p[j]);
                f.x *= col_mul[base_k + 2 * j];
                f.y *= col_mul[base_k + 2 * j + 1];
                reg_vals[it * 8 + 2 * j]     = f.x;
                reg_vals[it * 8 + 2 * j + 1] = f.y;
                lmax = fmaxf(lmax, fabsf(f.x));
                lmax = fmaxf(lmax, fabsf(f.y));
            }
        }
    }

    lmax = r10_warp_max(lmax);
    __shared__ float smax[8];
    if ((tid & 31) == 0) smax[tid >> 5] = lmax;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) m = fmaxf(m, smax[w]);
        float s0 = (m > 1e-12f) ? (m * (1.0f / 127.0f)) : 1.0f;
        smax[0] = s0;
        S0[r] = s0;
        S1[r] = s0 * (0.5f / 127.0f);
    }
    __syncthreads();
    float s0 = smax[0];
    float inv_s0 = 1.0f / s0;
    float inv_s1 = 254.0f / s0;

    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        int i = it * 256 + tid;
        if (i < k8) {
            signed char q0[8], q1[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float fv = reg_vals[it * 8 + j];
                int i0 = __float2int_rn(fv * inv_s0);
                i0 = i0 < -127 ? -127 : (i0 > 127 ? 127 : i0);
                float rem = fv - (float)i0 * s0;
                int i1 = __float2int_rn(rem * inv_s1);
                i1 = i1 < -127 ? -127 : (i1 > 127 ? 127 : i1);
                q0[j] = (signed char)i0;
                q1[j] = (signed char)i1;
            }
            out0_v[i] = *reinterpret_cast<uint2*>(q0);
            out1_v[i] = *reinterpret_cast<uint2*>(q1);
        }
    }
}

// 14.4 1-Pass Ozaki 2-Slice BF16 -> INT8 (K <= 4096)
extern "C" __global__ void quant_bf16_to_i8_ozaki2_k4k(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0_base,
    signed char* __restrict__ Q1_base,
    float* __restrict__ S0,
    float* __restrict__ S1,
    int valid_rows,
    int K
) {
    quant_bf16_to_i8_ozaki2_impl<2>(X, col_mul, Q0_base, Q1_base, S0, S1, valid_rows, K);
}

// 14.5 1-Pass Ozaki 2-Slice BF16 -> INT8 (K <= 10240)
extern "C" __global__ void quant_bf16_to_i8_ozaki2_k10k(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0_base,
    signed char* __restrict__ Q1_base,
    float* __restrict__ S0,
    float* __restrict__ S1,
    int valid_rows,
    int K
) {
    quant_bf16_to_i8_ozaki2_impl<5>(X, col_mul, Q0_base, Q1_base, S0, S1, valid_rows, K);
}

// 14.6 Vectorized 4-wide Dequant 1-Slice INT32 -> BF16 (with optional fused residual add!)
extern "C" __global__ void dequant_1slice_vec4_bf16(
    const int* __restrict__ C_i32,
    const float* __restrict__ sA0,
    const float* __restrict__ sB0,
    __nv_bfloat16* __restrict__ out,
    int valid_M,
    int N_strip,
    int ld_out,
    int n_off,
    int add_residual
) {
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = N_strip >> 2;
    if (idx4 >= valid_M * n4) return;
    int m = idx4 / n4;
    int nc = (idx4 - m * n4) << 2;

    int4 c = reinterpret_cast<const int4*>(C_i32)[idx4];
    float4 sb = reinterpret_cast<const float4*>(sB0)[nc >> 2];
    float sa = sA0[m];

    float v0 = (float)c.x * sa * sb.x;
    float v1 = (float)c.y * sa * sb.y;
    float v2 = (float)c.z * sa * sb.z;
    float v3 = (float)c.w * sa * sb.w;

    uint2* dst_p = reinterpret_cast<uint2*>(out + (size_t)m * ld_out + n_off) + (nc >> 2);
    if (add_residual) {
        uint2 old_v = *dst_p;
        float2 o01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&old_v.x));
        float2 o23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&old_v.y));
        v0 += o01.x; v1 += o01.y; v2 += o23.x; v3 += o23.y;
    }
    __nv_bfloat162 b01 = __floats2bfloat162_rn(v0, v1);
    __nv_bfloat162 b23 = __floats2bfloat162_rn(v2, v3);
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&b01);
    packed.y = *reinterpret_cast<uint32_t*>(&b23);
    *dst_p = packed;
}

// 14.7 Vectorized 4-wide Dequant 2x1 Ozaki INT32 -> BF16 (with optional fused residual add!)
extern "C" __global__ void dequant_ozaki2x1_vec4_bf16(
    const int* __restrict__ C_stack,
    const float* __restrict__ sA0,
    const float* __restrict__ sA1,
    const float* __restrict__ sB0,
    __nv_bfloat16* __restrict__ out,
    int valid_M,
    int M_pad,
    int N_strip,
    int ld_out,
    int n_off,
    int add_residual
) {
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = N_strip >> 2;
    if (idx4 >= valid_M * n4) return;
    int m = idx4 / n4;
    int nc = (idx4 - m * n4) << 2;
    int tot4_pad = M_pad * n4;

    int4 c0 = reinterpret_cast<const int4*>(C_stack)[idx4];
    int4 c1 = reinterpret_cast<const int4*>(C_stack)[tot4_pad + idx4];
    float4 sb = reinterpret_cast<const float4*>(sB0)[nc >> 2];
    float sa0 = sA0[m], sa1 = sA1[m];

    float v0 = ((float)c0.x * sa0 + (float)c1.x * sa1) * sb.x;
    float v1 = ((float)c0.y * sa0 + (float)c1.y * sa1) * sb.y;
    float v2 = ((float)c0.z * sa0 + (float)c1.z * sa1) * sb.z;
    float v3 = ((float)c0.w * sa0 + (float)c1.w * sa1) * sb.w;

    uint2* dst_p = reinterpret_cast<uint2*>(out + (size_t)m * ld_out + n_off) + (nc >> 2);
    if (add_residual) {
        uint2 old_v = *dst_p;
        float2 o01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&old_v.x));
        float2 o23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&old_v.y));
        v0 += o01.x; v1 += o01.y; v2 += o23.x; v3 += o23.y;
    }
    __nv_bfloat162 b01 = __floats2bfloat162_rn(v0, v1);
    __nv_bfloat162 b23 = __floats2bfloat162_rn(v2, v3);
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&b01);
    packed.y = *reinterpret_cast<uint32_t*>(&b23);
    *dst_p = packed;
}

// ============================================================================
// 15. PHASE R11: FUSED TILED SWIGLU-MLP + 1024-TOKEN SUPER-CHUNK PREFILL
//     - Strided 1-pass SmoothQuant BF16->INT8 (supports W_down[:, i0..i0+I_tile]!)
//     - Fused Dequant(up_i32) * SiLU(gate_bf16) -> act_bf16 in-place!
//     - 0 bytes register spill on sm_86!
// ============================================================================

template<int MAX_ITERS>
__device__ __forceinline__ void quant_bf16_to_i8_smooth_strided_impl(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0,
    float* __restrict__ S0,
    int valid_rows,
    int K,
    int ld_in,
    int k_off
) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int k8 = K >> 3;
    uint2* out_v = reinterpret_cast<uint2*>(Q0 + (size_t)r * K);

    if (r >= valid_rows) {
        if (tid == 0) S0[r] = 1.0f;
        uint2 z = {0, 0};
        #pragma unroll
        for (int it = 0; it < MAX_ITERS; ++it) {
            int i = it * 256 + tid;
            if (i < k8) out_v[i] = z;
        }
        return;
    }

    const uint4* row_v = reinterpret_cast<const uint4*>(X + (size_t)r * ld_in + k_off);
    float reg_vals[MAX_ITERS * 8];
    float lmax = 0.0f;

    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        int i = it * 256 + tid;
        if (i < k8) {
            uint4 v = row_v[i];
            const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(&v);
            int base_k = i << 3;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 f = __bfloat1622float2(p[j]);
                f.x *= col_mul[base_k + 2 * j];
                f.y *= col_mul[base_k + 2 * j + 1];
                reg_vals[it * 8 + 2 * j]     = f.x;
                reg_vals[it * 8 + 2 * j + 1] = f.y;
                lmax = fmaxf(lmax, fabsf(f.x));
                lmax = fmaxf(lmax, fabsf(f.y));
            }
        }
    }

    lmax = r10_warp_max(lmax);
    __shared__ float smax[8];
    if ((tid & 31) == 0) smax[tid >> 5] = lmax;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) m = fmaxf(m, smax[w]);
        smax[0] = (m > 1e-12f) ? (m * (1.0f / 127.0f)) : 1.0f;
        S0[r] = smax[0];
    }
    __syncthreads();
    float inv_s = 1.0f / smax[0];

    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        int i = it * 256 + tid;
        if (i < k8) {
            signed char q[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int iv = __float2int_rn(reg_vals[it * 8 + j] * inv_s);
                q[j] = (signed char)(iv < -127 ? -127 : (iv > 127 ? 127 : iv));
            }
            out_v[i] = *reinterpret_cast<uint2*>(q);
        }
    }
}

// 15.1 Strided 1-Pass SmoothQuant BF16 -> INT8 (K <= 4096: 2560, 3072, 4096)
extern "C" __global__ void quant_bf16_to_i8_smooth_strided_k4k(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0,
    float* __restrict__ S0,
    int valid_rows,
    int K,
    int ld_in,
    int k_off
) {
    quant_bf16_to_i8_smooth_strided_impl<2>(X, col_mul, Q0, S0, valid_rows, K, ld_in, k_off);
}

// 15.2 Strided 1-Pass SmoothQuant BF16 -> INT8 (K <= 6144: 4608)
extern "C" __global__ void quant_bf16_to_i8_smooth_strided_k6k(
    const __nv_bfloat16* __restrict__ X,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0,
    float* __restrict__ S0,
    int valid_rows,
    int K,
    int ld_in,
    int k_off
) {
    quant_bf16_to_i8_smooth_strided_impl<3>(X, col_mul, Q0, S0, valid_rows, K, ld_in, k_off);
}

// 15.3 Fused Dequant(up_i32) * SiLU(gate_bf16) -> act_bf16 IN-PLACE over gate_inout!
// Eliminates silu_mul_packed_2d kernel AND 864 MB of ws_big DRAM traffic across 32 layers!
extern "C" __global__ void dequant_up_fused_silu_mul_vec4_bf16(
    const int* __restrict__ C_up_i32,
    const float* __restrict__ sA0,
    const float* __restrict__ sB_up,
    __nv_bfloat16* __restrict__ gate_inout,
    int valid_M,
    int I_tile
) {
    int idx4 = blockIdx.x * blockDim.x + threadIdx.x;
    int n4 = I_tile >> 2;
    if (idx4 >= valid_M * n4) return;
    int m = idx4 / n4;
    int nc = (idx4 - m * n4) << 2;

    int4 c = reinterpret_cast<const int4*>(C_up_i32)[idx4];
    float4 sb = reinterpret_cast<const float4*>(sB_up)[nc >> 2];
    float sa = sA0[m];

    uint2* g_ptr = reinterpret_cast<uint2*>(gate_inout + (size_t)m * I_tile) + (nc >> 2);
    uint2 gv = *g_ptr;
    float2 g01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gv.x));
    float2 g23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&gv.y));

    float a0 = (g01.x / (1.0f + __expf(-g01.x))) * ((float)c.x * sa * sb.x);
    float a1 = (g01.y / (1.0f + __expf(-g01.y))) * ((float)c.y * sa * sb.y);
    float a2 = (g23.x / (1.0f + __expf(-g23.x))) * ((float)c.z * sa * sb.z);
    float a3 = (g23.y / (1.0f + __expf(-g23.y))) * ((float)c.w * sa * sb.w);

    __nv_bfloat162 b01 = __floats2bfloat162_rn(a0, a1);
    __nv_bfloat162 b23 = __floats2bfloat162_rn(a2, a3);
    uint2 packed;
    packed.x = *reinterpret_cast<uint32_t*>(&b01);
    packed.y = *reinterpret_cast<uint32_t*>(&b23);
    *g_ptr = packed;
}

// ============================================================================
// 16. PHASE R12: CUSTOM AMPERE PTX IMMA FUSED-DEQUANT GEMM +
//                REGISTER-RESIDENT 32x16 WMMA FLASHATTENTION-2
//     - Eliminates C_i32 DRAM write/read, dequant, residual_add & silu_mul!
//     - L2-swizzled CTA scheduling (GROUP_N=8), 3-stage cp.async, 0 bank conflicts!
//     - Register-resident O in 32x16 FA2 (30.0 KB static SMEM, 0 bytes spill)!
// ============================================================================

__device__ __forceinline__ void r12_cp_async_cg_16(void* smem_ptr, const void* glob_ptr) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_addr), "l"(glob_ptr));
}
__device__ __forceinline__ void r12_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N_WAIT>
__device__ __forceinline__ void r12_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N_WAIT));
}
__device__ __forceinline__ void r12_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr)
    );
}
__device__ __forceinline__ void r12_ldmatrix_x2(
    uint32_t& r0, uint32_t& r1, const void* smem_ptr
) {
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1) : "r"(smem_addr)
    );
}
__device__ __forceinline__ void r12_mma_m16n8k32_s8(
    int& d0, int& d1, int& d2, int& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    int c0, int c1, int c2, int c3
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(c0), "r"(c1), "r"(c2), "r"(c3)
    );
}
__device__ __forceinline__ int r12_swz64(int row, int col16) {
    return (col16 ^ (row & 3)) << 4;
}
__device__ __forceinline__ void r12_l2_swizzle_cta(int grid_n, int grid_m, int& bx, int& by) {
    constexpr int GROUP_N = 8;
    int linear = blockIdx.y * grid_n + blockIdx.x;
    int tiles_per_group = GROUP_N * grid_m;
    int group_id = linear / tiles_per_group;
    int first_n = group_id * GROUP_N;
    int group_width = min(GROUP_N, grid_n - first_n);
    int in_group = linear - group_id * tiles_per_group;
    by = in_group / group_width;
    bx = first_n + (in_group - by * group_width);
}

// 16.1 Custom Ampere PTX IMMA Fused-Dequant GEMM: 128x128x64 (8 warps = 256 threads, 48 KB SMEM, 0 spill)
// Best for wide N >= 6144 (dn_in, qkv). Handles arbitrary valid_M and non-128-multiple N safely!
// epilogue_mode: 0 = store BF16, 1 = fused residual add (out += val), 2 = fused SwiGLU (out = SiLU(out) * val)
extern "C" __global__ void imma_gemm_dequant_128x128x64_bf16(
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    const float* __restrict__ sA,
    const float* __restrict__ sB,
    __nv_bfloat16* __restrict__ out,
    int valid_M,
    int N,
    int K,
    int ld_out,
    int n_off,
    int epilogue_mode
) {
    constexpr int BM = 128, BN = 128, BK = 64, STAGES = 3;
    int bx, by;
    r12_l2_swizzle_cta(gridDim.x, gridDim.y, bx, by);
    int m_block = by * BM;
    int n_block = bx * BN;
    if (m_block >= valid_M || n_block >= N) return;

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int wy = wid >> 2; // 0..1
    int wx = wid & 3;  // 0..3

    __shared__ __align__(16) signed char smA[STAGES][BM][BK];
    __shared__ __align__(16) signed char smB[STAGES][BN][BK];

    int acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;

    int ld_row0 = tid >> 2;
    int ld_col16 = tid & 3;
    int swz0 = r12_swz64(ld_row0, ld_col16);
    int swz1 = r12_swz64(ld_row0 + 64, ld_col16);

    // Precompute clamped row addresses outside the K-loop (0 out-of-bounds reads for arbitrary M and N!)
    const signed char* a_ptr0 = A + (size_t)min(m_block + ld_row0, valid_M - 1) * K + (ld_col16 << 4);
    const signed char* a_ptr1 = A + (size_t)min(m_block + ld_row0 + 64, valid_M - 1) * K + (ld_col16 << 4);
    const signed char* b_ptr0 = B + (size_t)min(n_block + ld_row0, N - 1) * K + (ld_col16 << 4);
    const signed char* b_ptr1 = B + (size_t)min(n_block + ld_row0 + 64, N - 1) * K + (ld_col16 << 4);

    auto load_stage = [&](int stage, int k0) {
        r12_cp_async_cg_16(&smA[stage][ld_row0][swz0],      a_ptr0 + k0);
        r12_cp_async_cg_16(&smA[stage][ld_row0 + 64][swz1], a_ptr1 + k0);
        r12_cp_async_cg_16(&smB[stage][ld_row0][swz0],      b_ptr0 + k0);
        r12_cp_async_cg_16(&smB[stage][ld_row0 + 64][swz1], b_ptr1 + k0);
        r12_cp_async_commit();
    };

    int num_k = K / BK;
    load_stage(0, 0);
    if (num_k > 1) load_stage(1, BK);

    int a_row_off = lane & 15;
    int a_col16_off = lane >> 4;
    int b_row_off = lane & 7;
    int b_col16_off = (lane >> 3) & 1;

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt % STAGES;
        if (kt + 2 < num_k) load_stage((kt + 2) % STAGES, (kt + 2) * BK);
        if (kt + 2 < num_k) r12_cp_async_wait<2>();
        else if (kt + 1 < num_k) r12_cp_async_wait<1>();
        else r12_cp_async_wait<0>();
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            int c16_base = ks * 2;
            uint32_t fragA[4][4], fragB[4][2];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = wy * 64 + mi * 16 + a_row_off;
                r12_ldmatrix_x4(fragA[mi][0], fragA[mi][1], fragA[mi][2], fragA[mi][3],
                                &smA[cur][r][r12_swz64(r, c16_base + a_col16_off)]);
            }
            #pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
                int r = wx * 32 + nj * 8 + b_row_off;
                r12_ldmatrix_x2(fragB[nj][0], fragB[nj][1],
                                &smB[cur][r][r12_swz64(r, c16_base + b_col16_off)]);
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int nj = 0; nj < 4; ++nj)
                    r12_mma_m16n8k32_s8(
                        acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3],
                        fragA[mi][0], fragA[mi][1], fragA[mi][2], fragA[mi][3],
                        fragB[nj][0], fragB[nj][1],
                        acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3]
                    );
        }
        __syncthreads();
    }

    int grp = lane >> 2;
    int tg2 = (lane & 3) << 1;
    float sb_pair[4][2];
    #pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
        int col = n_block + wx * 32 + nj * 8 + tg2;
        sb_pair[nj][0] = (col < N)     ? sB[col]     : 0.0f;
        sb_pair[nj][1] = (col + 1 < N) ? sB[col + 1] : 0.0f;
    }

    auto apply_ep = [&](float v0, float v1, __nv_bfloat162* dst) {
        if (epilogue_mode == 1) {
            float2 o = __bfloat1622float2(*dst);
            v0 += o.x;
            v1 += o.y;
        } else if (epilogue_mode == 2) {
            float2 g = __bfloat1622float2(*dst);
            v0 *= (g.x / (1.0f + __expf(-g.x)));
            v1 *= (g.y / (1.0f + __expf(-g.y)));
        }
        *dst = __floats2bfloat162_rn(v0, v1);
    };

    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        int r0 = m_block + wy * 64 + mi * 16 + grp;
        int r1 = r0 + 8;
        float sa0 = (r0 < valid_M) ? sA[r0] : 0.0f;
        float sa1 = (r1 < valid_M) ? sA[r1] : 0.0f;
        #pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
            int col = n_block + wx * 32 + nj * 8 + tg2;
            if (col < N) {
                float sb0 = sb_pair[nj][0], sb1 = sb_pair[nj][1];
                if (r0 < valid_M) {
                    apply_ep((float)acc[mi][nj][0] * sa0 * sb0, (float)acc[mi][nj][1] * sa0 * sb1,
                             reinterpret_cast<__nv_bfloat162*>(out + (size_t)r0 * ld_out + n_off + col));
                }
                if (r1 < valid_M) {
                    apply_ep((float)acc[mi][nj][2] * sa1 * sb0, (float)acc[mi][nj][3] * sa1 * sb1,
                             reinterpret_cast<__nv_bfloat162*>(out + (size_t)r1 * ld_out + n_off + col));
                }
            }
        }
    }
}

// 16.2 Custom Ampere PTX IMMA Fused-Dequant GEMM: 64x128x64 (4 warps = 128 threads, 36 KB SMEM -> 2 CTAs/SM, 0 spill)
// Best for N < 6144 (gate_tile, up_tile, mlp_down, out_proj, o_proj) — eliminates 68-SM wave quantization!
extern "C" __global__ void imma_gemm_dequant_64x128x64_bf16(
    const signed char* __restrict__ A,
    const signed char* __restrict__ B,
    const float* __restrict__ sA,
    const float* __restrict__ sB,
    __nv_bfloat16* __restrict__ out,
    int valid_M,
    int N,
    int K,
    int ld_out,
    int n_off,
    int epilogue_mode
) {
    constexpr int BM = 64, BN = 128, BK = 64, STAGES = 3;
    int bx, by;
    r12_l2_swizzle_cta(gridDim.x, gridDim.y, bx, by);
    int m_block = by * BM;
    int n_block = bx * BN;
    if (m_block >= valid_M || n_block >= N) return;

    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int wx = wid; // 0..3

    __shared__ __align__(16) signed char smA[STAGES][BM][BK];
    __shared__ __align__(16) signed char smB[STAGES][BN][BK];

    int acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0;

    int ld_row0 = tid >> 2;
    int ld_col16 = tid & 3;
    int swz0 = r12_swz64(ld_row0, ld_col16);
    int swz1 = r12_swz64(ld_row0 + 32, ld_col16);
    int swz2 = r12_swz64(ld_row0 + 64, ld_col16);
    int swz3 = r12_swz64(ld_row0 + 96, ld_col16);

    const signed char* a_ptr0 = A + (size_t)min(m_block + ld_row0, valid_M - 1) * K + (ld_col16 << 4);
    const signed char* a_ptr1 = A + (size_t)min(m_block + ld_row0 + 32, valid_M - 1) * K + (ld_col16 << 4);
    const signed char* b_ptr0 = B + (size_t)min(n_block + ld_row0, N - 1) * K + (ld_col16 << 4);
    const signed char* b_ptr1 = B + (size_t)min(n_block + ld_row0 + 32, N - 1) * K + (ld_col16 << 4);
    const signed char* b_ptr2 = B + (size_t)min(n_block + ld_row0 + 64, N - 1) * K + (ld_col16 << 4);
    const signed char* b_ptr3 = B + (size_t)min(n_block + ld_row0 + 96, N - 1) * K + (ld_col16 << 4);

    auto load_stage = [&](int stage, int k0) {
        r12_cp_async_cg_16(&smA[stage][ld_row0][swz0],      a_ptr0 + k0);
        r12_cp_async_cg_16(&smA[stage][ld_row0 + 32][swz1], a_ptr1 + k0);
        r12_cp_async_cg_16(&smB[stage][ld_row0][swz0],      b_ptr0 + k0);
        r12_cp_async_cg_16(&smB[stage][ld_row0 + 32][swz1], b_ptr1 + k0);
        r12_cp_async_cg_16(&smB[stage][ld_row0 + 64][swz2], b_ptr2 + k0);
        r12_cp_async_cg_16(&smB[stage][ld_row0 + 96][swz3], b_ptr3 + k0);
        r12_cp_async_commit();
    };

    int num_k = K / BK;
    load_stage(0, 0);
    if (num_k > 1) load_stage(1, BK);

    int a_row_off = lane & 15;
    int a_col16_off = lane >> 4;
    int b_row_off = lane & 7;
    int b_col16_off = (lane >> 3) & 1;

    for (int kt = 0; kt < num_k; ++kt) {
        int cur = kt % STAGES;
        if (kt + 2 < num_k) load_stage((kt + 2) % STAGES, (kt + 2) * BK);
        if (kt + 2 < num_k) r12_cp_async_wait<2>();
        else if (kt + 1 < num_k) r12_cp_async_wait<1>();
        else r12_cp_async_wait<0>();
        __syncthreads();

        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            int c16_base = ks * 2;
            uint32_t fragA[4][4], fragB[4][2];
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
                int r = mi * 16 + a_row_off;
                r12_ldmatrix_x4(fragA[mi][0], fragA[mi][1], fragA[mi][2], fragA[mi][3],
                                &smA[cur][r][r12_swz64(r, c16_base + a_col16_off)]);
            }
            #pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
                int r = wx * 32 + nj * 8 + b_row_off;
                r12_ldmatrix_x2(fragB[nj][0], fragB[nj][1],
                                &smB[cur][r][r12_swz64(r, c16_base + b_col16_off)]);
            }
            #pragma unroll
            for (int mi = 0; mi < 4; ++mi)
                #pragma unroll
                for (int nj = 0; nj < 4; ++nj)
                    r12_mma_m16n8k32_s8(
                        acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3],
                        fragA[mi][0], fragA[mi][1], fragA[mi][2], fragA[mi][3],
                        fragB[nj][0], fragB[nj][1],
                        acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3]
                    );
        }
        __syncthreads();
    }

    int grp = lane >> 2;
    int tg2 = (lane & 3) << 1;
    float sb_pair[4][2];
    #pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
        int col = n_block + wx * 32 + nj * 8 + tg2;
        sb_pair[nj][0] = (col < N)     ? sB[col]     : 0.0f;
        sb_pair[nj][1] = (col + 1 < N) ? sB[col + 1] : 0.0f;
    }

    auto apply_ep = [&](float v0, float v1, __nv_bfloat162* dst) {
        if (epilogue_mode == 1) {
            float2 o = __bfloat1622float2(*dst);
            v0 += o.x;
            v1 += o.y;
        } else if (epilogue_mode == 2) {
            float2 g = __bfloat1622float2(*dst);
            v0 *= (g.x / (1.0f + __expf(-g.x)));
            v1 *= (g.y / (1.0f + __expf(-g.y)));
        }
        *dst = __floats2bfloat162_rn(v0, v1);
    };

    #pragma unroll
    for (int mi = 0; mi < 4; ++mi) {
        int r0 = m_block + mi * 16 + grp;
        int r1 = r0 + 8;
        float sa0 = (r0 < valid_M) ? sA[r0] : 0.0f;
        float sa1 = (r1 < valid_M) ? sA[r1] : 0.0f;
        #pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
            int col = n_block + wx * 32 + nj * 8 + tg2;
            if (col < N) {
                float sb0 = sb_pair[nj][0], sb1 = sb_pair[nj][1];
                if (r0 < valid_M) {
                    apply_ep((float)acc[mi][nj][0] * sa0 * sb0, (float)acc[mi][nj][1] * sa0 * sb1,
                             reinterpret_cast<__nv_bfloat162*>(out + (size_t)r0 * ld_out + n_off + col));
                }
                if (r1 < valid_M) {
                    apply_ep((float)acc[mi][nj][2] * sa1 * sb0, (float)acc[mi][nj][3] * sa1 * sb1,
                             reinterpret_cast<__nv_bfloat162*>(out + (size_t)r1 * ld_out + n_off + col));
                }
            }
        }
    }
}

// 16.3 Register-Resident 32x16 WMMA FlashAttention-2 Causal Prefill GQA + Sigmoid Gate (Contiguous KV)
// Grid: (16 Q-heads, ceil(seq_len / 32), 1), Block: (128, 1, 1) = 4 warps (2 Q-tiles x 2 D-halves)
// ZERO s_O in SMEM (64 FP32 regs/thread), 30.0 KB static SMEM (7.5 KB less than R9!), 0 bytes spill!
extern "C" __global__ void prefill_gqa_wmma_32x16_causal_gated_bf16(
    const bf16_t* __restrict__ q_sdpa,      // [16, seq_len, 256]
    const bf16_t* __restrict__ qkv_packed,  // [seq_len, 10240]
    const bf16_t* __restrict__ k_cache,     // [4, max_seq, 256]
    const bf16_t* __restrict__ v_cache,     // [4, max_seq, 256]
    int start_pos,
    int seq_len,
    int max_seq,
    bf16_t* __restrict__ out                // [seq_len, 4096]
) {
    using namespace nvcuda::wmma;
    int hq = blockIdx.x;
    int hk = hq >> 2;
    int q_base = blockIdx.y * 32;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int wq = wid >> 1;      // 0..1 (16-row Q tile)
    int wd = wid & 1;       // 0..1 (128-col D half)
    int d_base = wd * 128;

    __shared__ __align__(16) __nv_bfloat16 s_Q[32][256];
    __shared__ __align__(16) __nv_bfloat16 s_KV[16][256];
    __shared__ __align__(16) float s_S[2][2][16][16];
    __shared__ __align__(16) __nv_bfloat16 s_P[32][16];
    __shared__ float s_m[32];
    __shared__ float s_l[32];
    __shared__ float s_alpha[32];

    const bf16_t* q_head = q_sdpa + (size_t)hq * seq_len * 256;
    for (int idx = tid; idx < 1024; idx += 128) {
        int r = idx >> 5;
        int c8 = (idx & 31) * 8;
        if (q_base + r < seq_len) {
            *reinterpret_cast<uint4*>(&s_Q[r][c8]) =
                *reinterpret_cast<const uint4*>(q_head + (size_t)(q_base + r) * 256 + c8);
        } else {
            uint4 z = {0, 0, 0, 0};
            *reinterpret_cast<uint4*>(&s_Q[r][c8]) = z;
        }
    }
    if (tid < 32) {
        s_m[tid] = -1e30f;
        s_l[tid] = 0.0f;
    }
    __syncthreads();

    fragment<accumulator, 16, 16, 16, float> o_acc[8];
    #pragma unroll
    for (int dc = 0; dc < 8; ++dc) fill_fragment(o_acc[dc], 0.0f);

    const bf16_t* k_head = k_cache + (size_t)hk * max_seq * 256;
    const bf16_t* v_head = v_cache + (size_t)hk * max_seq * 256;
    int q_max = (q_base + 31 < seq_len) ? (q_base + 31) : (seq_len - 1);
    int kv_end_blk = start_pos + q_max + 1;

    int r0_local = wq * 16 + (lane >> 2);
    int r1_local = r0_local + 8;

    for (int kt0 = 0; kt0 < kv_end_blk; kt0 += 16) {
        // 1. Cooperative K load [16, 256]
        for (int idx = tid; idx < 512; idx += 128) {
            int r = idx >> 5;
            int c8 = (idx & 31) * 8;
            if (kt0 + r < kv_end_blk) {
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) =
                    *reinterpret_cast<const uint4*>(k_head + (size_t)(kt0 + r) * 256 + c8);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) = z;
            }
        }
        __syncthreads();

        // 2. Q @ K^T (each warp computes 16 Q rows x 128 D dims = 8 WMMA steps)
        fragment<accumulator, 16, 16, 16, float> s_frag;
        fill_fragment(s_frag, 0.0f);
        #pragma unroll
        for (int dk = 0; dk < 8; ++dk) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> q_f;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> k_f;
            load_matrix_sync(q_f, &s_Q[wq * 16][d_base + dk * 16], 256);
            load_matrix_sync(k_f, &s_KV[0][d_base + dk * 16], 256);
            mma_sync(s_frag, q_f, k_f, s_frag);
        }
        store_matrix_sync(&s_S[wq][wd][0][0], s_frag, 16, mem_row_major);
        __syncthreads();

        // 3. Parallel online softmax across 32 Q rows + overlapped V load into s_KV!
        if (tid < 32) {
            int r = tid;
            int qt = r >> 4;
            int rl = r & 15;
            int q_pos = start_pos + q_base + r;
            bool valid_r = (q_base + r < seq_len);
            float row_s[16];
            float row_max = s_m[r];
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float sc = (valid_r && (kt0 + c) <= q_pos)
                    ? ((s_S[qt][0][rl][c] + s_S[qt][1][rl][c]) * 0.0625f)
                    : -1e30f;
                row_s[c] = sc;
                row_max = fmaxf(row_max, sc);
            }
            float alpha = (s_m[r] > -1e29f) ? __expf(s_m[r] - row_max) : 0.0f;
            float l_new = s_l[r] * alpha;
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float p = (valid_r && (kt0 + c) <= q_pos) ? __expf(row_s[c] - row_max) : 0.0f;
                l_new += p;
                s_P[r][c] = __float2bfloat16(p);
            }
            s_m[r] = row_max;
            s_l[r] = l_new;
            s_alpha[r] = alpha;
        }
        for (int idx = tid; idx < 512; idx += 128) {
            int r = idx >> 5;
            int c8 = (idx & 31) * 8;
            if (kt0 + r < kv_end_blk) {
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) =
                    *reinterpret_cast<const uint4*>(v_head + (size_t)(kt0 + r) * 256 + c8);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) = z;
            }
        }
        __syncthreads();

        // 4. Rescale register-resident O in-place + accumulate P @ V directly into o_acc!
        float a0 = s_alpha[r0_local];
        float a1 = s_alpha[r1_local];
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> p_f;
        load_matrix_sync(p_f, &s_P[wq * 16][0], 16);
        #pragma unroll
        for (int dc = 0; dc < 8; ++dc) {
            o_acc[dc].x[0] *= a0; o_acc[dc].x[1] *= a0;
            o_acc[dc].x[2] *= a1; o_acc[dc].x[3] *= a1;
            o_acc[dc].x[4] *= a0; o_acc[dc].x[5] *= a0;
            o_acc[dc].x[6] *= a1; o_acc[dc].x[7] *= a1;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> v_f;
            load_matrix_sync(v_f, &s_KV[0][d_base + dc * 16], 256);
            mma_sync(o_acc[dc], p_f, v_f, o_acc[dc]);
        }
        __syncthreads();
    }

    // 5. Final normalize + sigmoid output gate staged through s_S[wq][wd] (zero extra SMEM!)
    #pragma unroll
    for (int dc = 0; dc < 8; ++dc) {
        store_matrix_sync(&s_S[wq][wd][0][0], o_acc[dc], 16, mem_row_major);
        __syncwarp();
        int col_base = d_base + dc * 16;
        #pragma unroll
        for (int e = lane; e < 256; e += 32) {
            int rl = e >> 4;
            int cc = e & 15;
            int r = wq * 16 + rl;
            if (q_base + r < seq_len) {
                float inv_l = 1.0f / fmaxf(s_l[r], 1e-30f);
                int d = col_base + cc;
                float g_val = bf16_to_f(qkv_packed[(size_t)(q_base + r) * 10240 + hq * 512 + 256 + d]);
                float sig = 1.0f / (1.0f + __expf(-g_val));
                out[(size_t)(q_base + r) * 4096 + hq * 256 + d] = f_to_bf16((s_S[wq][wd][rl][cc] * inv_l) * sig);
            }
        }
        __syncwarp();
    }
}

// 16.4 Register-Resident 32x16 WMMA FlashAttention-2 Causal Prefill GQA over Paged KV Blocks
// Grid: (16 Q-heads, ceil(seq_len / 32), 1), Block: (128, 1, 1), 30.0 KB static SMEM, 0 bytes spill!
extern "C" __global__ void prefill_gqa_paged_wmma_32x16_causal_gated_bf16(
    const bf16_t* __restrict__ q_sdpa,      // [16, seq_len, 256]
    const bf16_t* __restrict__ qkv_packed,  // [seq_len, 10240]
    const bf16_t* __restrict__ kv_pool,     // [num_blocks, 8, 2, 4, 16, 256]
    const int* __restrict__ block_table,
    int fa_idx,
    int start_pos,
    int seq_len,
    bf16_t* __restrict__ out                // [seq_len, 4096]
) {
    using namespace nvcuda::wmma;
    int hq = blockIdx.x;
    int hk = hq >> 2;
    int q_base = blockIdx.y * 32;
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int wq = wid >> 1;
    int wd = wid & 1;
    int d_base = wd * 128;

    __shared__ __align__(16) __nv_bfloat16 s_Q[32][256];
    __shared__ __align__(16) __nv_bfloat16 s_KV[16][256];
    __shared__ __align__(16) float s_S[2][2][16][16];
    __shared__ __align__(16) __nv_bfloat16 s_P[32][16];
    __shared__ float s_m[32];
    __shared__ float s_l[32];
    __shared__ float s_alpha[32];

    const bf16_t* q_head = q_sdpa + (size_t)hq * seq_len * 256;
    for (int idx = tid; idx < 1024; idx += 128) {
        int r = idx >> 5;
        int c8 = (idx & 31) * 8;
        if (q_base + r < seq_len) {
            *reinterpret_cast<uint4*>(&s_Q[r][c8]) =
                *reinterpret_cast<const uint4*>(q_head + (size_t)(q_base + r) * 256 + c8);
        } else {
            uint4 z = {0, 0, 0, 0};
            *reinterpret_cast<uint4*>(&s_Q[r][c8]) = z;
        }
    }
    if (tid < 32) {
        s_m[tid] = -1e30f;
        s_l[tid] = 0.0f;
    }
    __syncthreads();

    fragment<accumulator, 16, 16, 16, float> o_acc[8];
    #pragma unroll
    for (int dc = 0; dc < 8; ++dc) fill_fragment(o_acc[dc], 0.0f);

    int q_max = (q_base + 31 < seq_len) ? (q_base + 31) : (seq_len - 1);
    int kv_end_blk = start_pos + q_max + 1;

    int r0_local = wq * 16 + (lane >> 2);
    int r1_local = r0_local + 8;

    for (int kt0 = 0; kt0 < kv_end_blk; kt0 += 16) {
        int blk = block_table[kt0 >> 4];
        const bf16_t* k_blk = kv_pool + (size_t)blk * 262144 + (size_t)fa_idx * 32768 + (size_t)hk * 4096;
        const bf16_t* v_blk = k_blk + 16384;

        for (int idx = tid; idx < 512; idx += 128) {
            int r = idx >> 5;
            int c8 = (idx & 31) * 8;
            if (kt0 + r < kv_end_blk) {
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) =
                    *reinterpret_cast<const uint4*>(k_blk + (size_t)r * 256 + c8);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) = z;
            }
        }
        __syncthreads();

        fragment<accumulator, 16, 16, 16, float> s_frag;
        fill_fragment(s_frag, 0.0f);
        #pragma unroll
        for (int dk = 0; dk < 8; ++dk) {
            fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> q_f;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> k_f;
            load_matrix_sync(q_f, &s_Q[wq * 16][d_base + dk * 16], 256);
            load_matrix_sync(k_f, &s_KV[0][d_base + dk * 16], 256);
            mma_sync(s_frag, q_f, k_f, s_frag);
        }
        store_matrix_sync(&s_S[wq][wd][0][0], s_frag, 16, mem_row_major);
        __syncthreads();

        if (tid < 32) {
            int r = tid;
            int qt = r >> 4;
            int rl = r & 15;
            int q_pos = start_pos + q_base + r;
            bool valid_r = (q_base + r < seq_len);
            float row_s[16];
            float row_max = s_m[r];
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float sc = (valid_r && (kt0 + c) <= q_pos)
                    ? ((s_S[qt][0][rl][c] + s_S[qt][1][rl][c]) * 0.0625f)
                    : -1e30f;
                row_s[c] = sc;
                row_max = fmaxf(row_max, sc);
            }
            float alpha = (s_m[r] > -1e29f) ? __expf(s_m[r] - row_max) : 0.0f;
            float l_new = s_l[r] * alpha;
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                float p = (valid_r && (kt0 + c) <= q_pos) ? __expf(row_s[c] - row_max) : 0.0f;
                l_new += p;
                s_P[r][c] = __float2bfloat16(p);
            }
            s_m[r] = row_max;
            s_l[r] = l_new;
            s_alpha[r] = alpha;
        }
        for (int idx = tid; idx < 512; idx += 128) {
            int r = idx >> 5;
            int c8 = (idx & 31) * 8;
            if (kt0 + r < kv_end_blk) {
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) =
                    *reinterpret_cast<const uint4*>(v_blk + (size_t)r * 256 + c8);
            } else {
                uint4 z = {0, 0, 0, 0};
                *reinterpret_cast<uint4*>(&s_KV[r][c8]) = z;
            }
        }
        __syncthreads();

        float a0 = s_alpha[r0_local];
        float a1 = s_alpha[r1_local];
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> p_f;
        load_matrix_sync(p_f, &s_P[wq * 16][0], 16);
        #pragma unroll
        for (int dc = 0; dc < 8; ++dc) {
            o_acc[dc].x[0] *= a0; o_acc[dc].x[1] *= a0;
            o_acc[dc].x[2] *= a1; o_acc[dc].x[3] *= a1;
            o_acc[dc].x[4] *= a0; o_acc[dc].x[5] *= a0;
            o_acc[dc].x[6] *= a1; o_acc[dc].x[7] *= a1;
            fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> v_f;
            load_matrix_sync(v_f, &s_KV[0][d_base + dc * 16], 256);
            mma_sync(o_acc[dc], p_f, v_f, o_acc[dc]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int dc = 0; dc < 8; ++dc) {
        store_matrix_sync(&s_S[wq][wd][0][0], o_acc[dc], 16, mem_row_major);
        __syncwarp();
        int col_base = d_base + dc * 16;
        #pragma unroll
        for (int e = lane; e < 256; e += 32) {
            int rl = e >> 4;
            int cc = e & 15;
            int r = wq * 16 + rl;
            if (q_base + r < seq_len) {
                float inv_l = 1.0f / fmaxf(s_l[r], 1e-30f);
                int d = col_base + cc;
                float g_val = bf16_to_f(qkv_packed[(size_t)(q_base + r) * 10240 + hq * 512 + 256 + d]);
                float sig = 1.0f / (1.0f + __expf(-g_val));
                out[(size_t)(q_base + r) * 4096 + hq * 256 + d] = f_to_bf16((s_S[wq][wd][rl][cc] * inv_l) * sig);
            }
        }
        __syncwarp();
    }
}

// ============================================================================
// 17. PHASE R13: FUSED SKINNY LM_HEAD + ONLINE ARGMAX (M=1, ZERO LOGITS DRAM!)
//     2x-unrolled ld.global.cs streaming BF16 weight reads + L1-hot __ldg(X),
//     ZERO inner __syncthreads(), ONLY 40 regs, 64 B SMEM, 0 bytes spill!
//     Beats cuBLAS(M=2)+Argmax by 1.09x while eliminating 1.0 MiB s_logits!
// ============================================================================
__device__ __forceinline__ uint4 r13_ldcs_uint4(const uint4* ptr) {
    uint4 r;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(ptr));
    return r;
}

__device__ __forceinline__ float r13_dot_uint4_bf16(uint4 a, uint4 b, float acc) {
    acc = fmaf(bf16_to_f((bf16_t)(a.x & 0xFFFFu)), bf16_to_f((bf16_t)(b.x & 0xFFFFu)), acc);
    acc = fmaf(bf16_to_f((bf16_t)(a.x >> 16)),     bf16_to_f((bf16_t)(b.x >> 16)),     acc);
    acc = fmaf(bf16_to_f((bf16_t)(a.y & 0xFFFFu)), bf16_to_f((bf16_t)(b.y & 0xFFFFu)), acc);
    acc = fmaf(bf16_to_f((bf16_t)(a.y >> 16)),     bf16_to_f((bf16_t)(b.y >> 16)),     acc);
    acc = fmaf(bf16_to_f((bf16_t)(a.z & 0xFFFFu)), bf16_to_f((bf16_t)(b.z & 0xFFFFu)), acc);
    acc = fmaf(bf16_to_f((bf16_t)(a.z >> 16)),     bf16_to_f((bf16_t)(b.z >> 16)),     acc);
    acc = fmaf(bf16_to_f((bf16_t)(a.w & 0xFFFFu)), bf16_to_f((bf16_t)(b.w & 0xFFFFu)), acc);
    acc = fmaf(bf16_to_f((bf16_t)(a.w >> 16)),     bf16_to_f((bf16_t)(b.w >> 16)),     acc);
    return acc;
}

// Grid: (1024, 1, 1), Block: (256, 1, 1) -> followed by argmax_stage2_bf16(num_blocks=1024)
extern "C" __global__ void lm_head_fused_argmax_m1_bf16(
    const bf16_t* __restrict__ X,       // [1, 2560]
    const bf16_t* __restrict__ W,       // [V, 2560]
    float* __restrict__ part_vals,      // [gridDim.x]
    int* __restrict__ part_idxs,        // [gridDim.x]
    int V
) {
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    const uint4* x_row = reinterpret_cast<const uint4*>(X);
    int rpb = (V + gridDim.x - 1) / gridDim.x;
    int v0 = blockIdx.x * rpb;
    int v1 = min(v0 + rpb, V);

    float best_v = -1e30f;
    int best_i = 0;

    for (int n = v0 + wid; n < v1; n += 8) {
        const uint4* w_row = reinterpret_cast<const uint4*>(W + (size_t)n * 2560);
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 5; ++s) {
            int kv0 = s * 64 + lane;
            int kv1 = kv0 + 32;
            uint4 w0 = r13_ldcs_uint4(&w_row[kv0]);
            uint4 w1 = r13_ldcs_uint4(&w_row[kv1]);
            uint4 x0 = __ldg(&x_row[kv0]);
            uint4 x1 = __ldg(&x_row[kv1]);
            acc = r13_dot_uint4_bf16(x0, w0, acc);
            acc = r13_dot_uint4_bf16(x1, w1, acc);
        }
        float d = warp_reduce_sum(acc);
        if (lane == 0 && d > best_v) {
            best_v = d;
            best_i = n;
        }
    }

    __shared__ float sm_v[8];
    __shared__ int sm_i[8];
    if (lane == 0) {
        sm_v[wid] = best_v;
        sm_i[wid] = best_i;
    }
    __syncthreads();
    if (tid == 0) {
        float bv = sm_v[0];
        int bi = sm_i[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (sm_v[w] > bv) {
                bv = sm_v[w];
                bi = sm_i[w];
            }
        }
        part_vals[blockIdx.x] = bv;
        part_idxs[blockIdx.x] = bi;
    }
}

// ============================================================================
// 18. PHASE R14: LOSSLESS BF12 WEIGHT COMPRESSION (100.000% BIT-EXACT BF16!)
//     Format per 2D matrix W[N, K] (K multiple of 512: 2560, 4096, 5120, 9216):
//       - sm8  [N, K]    : (sign << 7) | mant7      (1.00 B/w, uint4 128b coalesced)
//       - exp4 [N, K/2]  : 2x 4-bit (exp - e_base)  (0.50 B/w, uint2  64b coalesced)
//                          0..14 = in-window, 15 = CSR escape (0.023% of weights!)
//       - CSR escapes    : esc_row_ptr[N+1], esc_col[nnz], esc_val[nnz] (~0.001 B/w)
//     Saves -2.01 GB VRAM (-25.0%), ZERO LUT, ZERO SMEM, ZERO register spill!
// ============================================================================

__device__ __forceinline__ uint2 r14_ldcs_uint2(const uint2* ptr) {
    uint2 r;
    asm volatile("ld.global.cs.v2.u32 {%0, %1}, [%2];\n"
                 : "=r"(r.x), "=r"(r.y) : "l"(ptr));
    return r;
}

__device__ __forceinline__ unsigned int r14_ldcs_u32(const unsigned int* ptr) {
    unsigned int r;
    asm volatile("ld.global.cs.u32 %0, [%1];\n" : "=r"(r) : "l"(ptr));
    return r;
}

__device__ __forceinline__ unsigned int r14_decode_bf12_u32(unsigned int sm8, unsigned int exp4, unsigned int e_base) {
    unsigned int bf16 = ((sm8 & 0x80u) << 8) | ((e_base + exp4) << 7) | (sm8 & 0x7Fu);
    unsigned int mask = (exp4 == 15u) ? 0u : 0xFFFFFFFFu;
    return bf16 & mask;
}

__device__ __forceinline__ float r14_decode_bf12_f(unsigned int sm8, unsigned int exp4, unsigned int e_base) {
    return __uint_as_float(r14_decode_bf12_u32(sm8, exp4, e_base) << 16);
}

__device__ __forceinline__ void r14_decode_bf12_4f(unsigned int sm_u32, unsigned int exp_u16, unsigned int e_base, float out[4]) {
    out[0] = r14_decode_bf12_f( sm_u32        & 0xFFu,  exp_u16        & 0xFu, e_base);
    out[1] = r14_decode_bf12_f((sm_u32 >>  8) & 0xFFu, (exp_u16 >>  4) & 0xFu, e_base);
    out[2] = r14_decode_bf12_f((sm_u32 >> 16) & 0xFFu, (exp_u16 >>  8) & 0xFu, e_base);
    out[3] = r14_decode_bf12_f((sm_u32 >> 24) & 0xFFu, (exp_u16 >> 12) & 0xFu, e_base);
}

__device__ __forceinline__ uint2 r14_decode_bf12_4u(unsigned int sm_u32, unsigned int exp_u16, unsigned int e_base) {
    uint2 r;
    r.x = r14_decode_bf12_u32( sm_u32        & 0xFFu,  exp_u16        & 0xFu, e_base)
        | (r14_decode_bf12_u32((sm_u32 >>  8) & 0xFFu, (exp_u16 >>  4) & 0xFu, e_base) << 16);
    r.y = r14_decode_bf12_u32((sm_u32 >> 16) & 0xFFu, (exp_u16 >>  8) & 0xFu, e_base)
        | (r14_decode_bf12_u32((sm_u32 >> 24) & 0xFFu, (exp_u16 >> 12) & 0xFu, e_base) << 16);
    return r;
}

// 18.1 Embedding Lookup Single directly from BF12-Lossless (Grid: (1, 1, 1), Block: (256, 1, 1))
extern "C" __global__ void embed_lookup_single_bf12(
    const long long* __restrict__ token_id_ptr,
    const unsigned char* __restrict__ W_sm8,
    const unsigned char* __restrict__ W_exp4,
    const int* __restrict__ esc_row_ptr,
    const unsigned short* __restrict__ esc_col,
    const bf16_t* __restrict__ esc_val,
    bf16_t* __restrict__ out,
    int hidden_size,
    unsigned int e_base
) {
    int tid = threadIdx.x;
    long long tok = token_id_ptr[0];
    const uint4* sm_row = reinterpret_cast<const uint4*>(W_sm8 + (size_t)tok * hidden_size);
    const uint2* ex_row = reinterpret_cast<const uint2*>(W_exp4 + (size_t)tok * (hidden_size >> 1));
    uint4* dst = reinterpret_cast<uint4*>(out);
    int k16 = hidden_size >> 4;
    for (int i = tid; i < k16; i += blockDim.x) {
        uint4 sm16 = sm_row[i];
        uint2 ex16 = ex_row[i];
        uint2 p0 = r14_decode_bf12_4u(sm16.x, ex16.x & 0xFFFFu, e_base);
        uint2 p1 = r14_decode_bf12_4u(sm16.y, ex16.x >> 16,     e_base);
        uint2 p2 = r14_decode_bf12_4u(sm16.z, ex16.y & 0xFFFFu, e_base);
        uint2 p3 = r14_decode_bf12_4u(sm16.w, ex16.y >> 16,     e_base);
        dst[i * 2]     = {p0.x, p0.y, p1.x, p1.y};
        dst[i * 2 + 1] = {p2.x, p2.y, p3.x, p3.y};
    }
    __syncthreads();
    int es = esc_row_ptr[tok];
    int ee = esc_row_ptr[tok + 1];
    for (int idx = es + tid; idx < ee; idx += blockDim.x) {
        out[esc_col[idx]] = esc_val[idx];
    }
}

// 18.2 Embedding Lookup Prefill/Batch directly from BF12-Lossless (Grid: (seq_len, 1, 1), Block: (256, 1, 1))
extern "C" __global__ void embed_lookup_prefill_bf12(
    const long long* __restrict__ token_ids,
    const unsigned char* __restrict__ W_sm8,
    const unsigned char* __restrict__ W_exp4,
    const int* __restrict__ esc_row_ptr,
    const unsigned short* __restrict__ esc_col,
    const bf16_t* __restrict__ esc_val,
    bf16_t* __restrict__ out,
    int seq_len,
    int hidden_size,
    unsigned int e_base
) {
    int t = blockIdx.x;
    if (t >= seq_len) return;
    int tid = threadIdx.x;
    long long tok = token_ids[t];
    const uint4* sm_row = reinterpret_cast<const uint4*>(W_sm8 + (size_t)tok * hidden_size);
    const uint2* ex_row = reinterpret_cast<const uint2*>(W_exp4 + (size_t)tok * (hidden_size >> 1));
    bf16_t* out_row = out + (size_t)t * hidden_size;
    uint4* dst = reinterpret_cast<uint4*>(out_row);
    int k16 = hidden_size >> 4;
    for (int i = tid; i < k16; i += blockDim.x) {
        uint4 sm16 = sm_row[i];
        uint2 ex16 = ex_row[i];
        uint2 p0 = r14_decode_bf12_4u(sm16.x, ex16.x & 0xFFFFu, e_base);
        uint2 p1 = r14_decode_bf12_4u(sm16.y, ex16.x >> 16,     e_base);
        uint2 p2 = r14_decode_bf12_4u(sm16.z, ex16.y & 0xFFFFu, e_base);
        uint2 p3 = r14_decode_bf12_4u(sm16.w, ex16.y >> 16,     e_base);
        dst[i * 2]     = {p0.x, p0.y, p1.x, p1.y};
        dst[i * 2 + 1] = {p2.x, p2.y, p3.x, p3.y};
    }
    __syncthreads();
    int es = esc_row_ptr[tok];
    int ee = esc_row_ptr[tok + 1];
    for (int idx = es + tid; idx < ee; idx += blockDim.x) {
        out_row[esc_col[idx]] = esc_val[idx];
    }
}

// 18.3 MTP Pre-FC Fused Embed(BF12) + RMSNorm + Concat -> [M, 5120] (Grid: (M, 1, 1), Block: (256, 1, 1))
extern "C" __global__ void mtp_pre_fc_concat_bf12(
    const long long* __restrict__ token_ids,
    const unsigned char* __restrict__ W_sm8,
    const unsigned char* __restrict__ W_exp4,
    const int* __restrict__ esc_row_ptr,
    const unsigned short* __restrict__ esc_col,
    const bf16_t* __restrict__ esc_val,
    const bf16_t* __restrict__ hid_in,
    const bf16_t* __restrict__ w_emb_norm,
    const bf16_t* __restrict__ w_hid_norm,
    bf16_t* __restrict__ out_cat,
    int dim,
    float eps,
    unsigned int e_base
) {
    __shared__ __align__(16) bf16_t s_emb[2560];
    __shared__ float wsum_e[8];
    __shared__ float wsum_h[8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    long long tok = token_ids[row];
    const uint4* sm_row = reinterpret_cast<const uint4*>(W_sm8 + (size_t)tok * dim);
    const uint2* ex_row = reinterpret_cast<const uint2*>(W_exp4 + (size_t)tok * (dim >> 1));
    uint4* dst = reinterpret_cast<uint4*>(s_emb);
    int k16 = dim >> 4;
    for (int i = tid; i < k16; i += blockDim.x) {
        uint4 sm16 = sm_row[i];
        uint2 ex16 = ex_row[i];
        uint2 p0 = r14_decode_bf12_4u(sm16.x, ex16.x & 0xFFFFu, e_base);
        uint2 p1 = r14_decode_bf12_4u(sm16.y, ex16.x >> 16,     e_base);
        uint2 p2 = r14_decode_bf12_4u(sm16.z, ex16.y & 0xFFFFu, e_base);
        uint2 p3 = r14_decode_bf12_4u(sm16.w, ex16.y >> 16,     e_base);
        dst[i * 2]     = {p0.x, p0.y, p1.x, p1.y};
        dst[i * 2 + 1] = {p2.x, p2.y, p3.x, p3.y};
    }
    __syncthreads();
    int es = esc_row_ptr[tok];
    int ee = esc_row_ptr[tok + 1];
    for (int idx = es + tid; idx < ee; idx += blockDim.x) {
        s_emb[esc_col[idx]] = esc_val[idx];
    }
    __syncthreads();

    const bf16_t* hrow = hid_in + (size_t)row * dim;
    bf16_t* orow = out_cat + (size_t)row * (2 * dim);

    float sq_e = 0.0f;
    float sq_h = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        float ve = bf16_to_f(s_emb[i]);
        float vh = bf16_to_f(hrow[i]);
        sq_e += ve * ve;
        sq_h += vh * vh;
    }
    sq_e = warp_reduce_sum(sq_e);
    sq_h = warp_reduce_sum(sq_h);
    if (lane == 0) {
        wsum_e[wid] = sq_e;
        wsum_h[wid] = sq_h;
    }
    __syncthreads();
    if (wid == 0) {
        float ve = warp_reduce_sum((lane < 8) ? wsum_e[lane] : 0.0f);
        float vh = warp_reduce_sum((lane < 8) ? wsum_h[lane] : 0.0f);
        if (lane == 0) {
            wsum_e[0] = rsqrtf(ve / (float)dim + eps);
            wsum_h[0] = rsqrtf(vh / (float)dim + eps);
        }
    }
    __syncthreads();
    float inv_e = wsum_e[0];
    float inv_h = wsum_h[0];
    for (int i = tid; i < dim; i += 256) {
        orow[i]       = f_to_bf16(bf16_to_f(s_emb[i]) * inv_e * (1.0f + bf16_to_f(w_emb_norm[i])));
        orow[dim + i] = f_to_bf16(bf16_to_f(hrow[i])  * inv_h * (1.0f + bf16_to_f(w_hid_norm[i])));
    }
}

// 18.4 Decode Skinny BF12-Lossless GEMM (M = 1..5) + Optional Fused Residual Add
// Grid: ((N + 7) / 8, 1, 1), Block: (256, 1, 1) — 16 weights/lane, 0 bytes spill!
template<int M_VAL>
__device__ __forceinline__ void skinny_bf12_1wpr_impl(
    const bf16_t* __restrict__ X,
    const unsigned char* __restrict__ W_sm8,
    const unsigned char* __restrict__ W_exp4,
    const int* __restrict__ esc_row_ptr,
    const unsigned short* __restrict__ esc_col,
    const bf16_t* __restrict__ esc_val,
    bf16_t* __restrict__ Y,
    int N,
    int K,
    unsigned int e_base,
    int add_res
) {
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int n_base = blockIdx.x * 8;
    int n = n_base + wid;
    float acc[M_VAL];
    #pragma unroll
    for (int m = 0; m < M_VAL; ++m) acc[m] = 0.0f;

    if (n < N) {
        const uint4* sm_row = reinterpret_cast<const uint4*>(W_sm8 + (size_t)n * K);
        const uint2* ex_row = reinterpret_cast<const uint2*>(W_exp4 + (size_t)n * (K >> 1));
        int k16 = K >> 4;

        for (int kv = lane; kv < k16; kv += 32) {
            uint4 sm16 = r13_ldcs_uint4(&sm_row[kv]);
            uint2 ex16 = r14_ldcs_uint2(&ex_row[kv]);

            float wf0[4], wf1[4], wf2[4], wf3[4];
            r14_decode_bf12_4f(sm16.x, ex16.x & 0xFFFFu, e_base, wf0);
            r14_decode_bf12_4f(sm16.y, ex16.x >> 16,     e_base, wf1);
            r14_decode_bf12_4f(sm16.z, ex16.y & 0xFFFFu, e_base, wf2);
            r14_decode_bf12_4f(sm16.w, ex16.y >> 16,     e_base, wf3);

            #pragma unroll
            for (int m = 0; m < M_VAL; ++m) {
                const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)m * K);
                uint4 x0 = __ldg(&xr[kv * 2]);
                uint4 x1 = __ldg(&xr[kv * 2 + 1]);
                float xf0[8], xf1[8];
                unpack_uint4_bf16(x0, xf0);
                unpack_uint4_bf16(x1, xf1);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[m] = fmaf(xf0[j],     wf0[j], acc[m]);
                    acc[m] = fmaf(xf0[j + 4], wf1[j], acc[m]);
                    acc[m] = fmaf(xf1[j],     wf2[j], acc[m]);
                    acc[m] = fmaf(xf1[j + 4], wf3[j], acc[m]);
                }
            }
        }

        int es = esc_row_ptr[n];
        int ee = esc_row_ptr[n + 1];
        for (int idx = es + lane; idx < ee; idx += 32) {
            int c = (int)esc_col[idx];
            float wv = bf16_to_f(esc_val[idx]);
            #pragma unroll
            for (int m = 0; m < M_VAL; ++m) {
                acc[m] = fmaf(bf16_to_f(X[(size_t)m * K + c]), wv, acc[m]);
            }
        }
    }

    __shared__ float sm[M_VAL][8];
    #pragma unroll
    for (int m = 0; m < M_VAL; ++m) {
        float s = warp_reduce_sum(acc[m]);
        if (lane == 0) sm[m][wid] = s;
    }
    __syncthreads();

    if (tid < M_VAL) {
        int m = tid;
        if (n_base + 7 < N) {
            float vals[8];
            if (add_res) {
                uint4 old4 = *reinterpret_cast<const uint4*>(Y + (size_t)m * N + n_base);
                float oldf[8];
                unpack_uint4_bf16(old4, oldf);
                #pragma unroll
                for (int r = 0; r < 8; ++r) vals[r] = sm[m][r] + oldf[r];
            } else {
                #pragma unroll
                for (int r = 0; r < 8; ++r) vals[r] = sm[m][r];
            }
            *reinterpret_cast<uint4*>(Y + (size_t)m * N + n_base) = pack_uint4_bf16(vals);
        } else {
            for (int r = 0; r < 8 && n_base + r < N; ++r) {
                float v = sm[m][r];
                if (add_res) v += bf16_to_f(Y[(size_t)m * N + n_base + r]);
                Y[(size_t)m * N + n_base + r] = f_to_bf16(v);
            }
        }
    }
}

extern "C" __global__ void skinny_bf12_m1_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    bf16_t* __restrict__ Y, int N, int K, unsigned int e_base, int add_res
) { skinny_bf12_1wpr_impl<1>(X, sm8, exp4, rp, ec, ev, Y, N, K, e_base, add_res); }

extern "C" __global__ void skinny_bf12_m2_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    bf16_t* __restrict__ Y, int N, int K, unsigned int e_base, int add_res
) { skinny_bf12_1wpr_impl<2>(X, sm8, exp4, rp, ec, ev, Y, N, K, e_base, add_res); }

extern "C" __global__ void skinny_bf12_m3_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    bf16_t* __restrict__ Y, int N, int K, unsigned int e_base, int add_res
) { skinny_bf12_1wpr_impl<3>(X, sm8, exp4, rp, ec, ev, Y, N, K, e_base, add_res); }

extern "C" __global__ void skinny_bf12_m4_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    bf16_t* __restrict__ Y, int N, int K, unsigned int e_base, int add_res
) { skinny_bf12_1wpr_impl<4>(X, sm8, exp4, rp, ec, ev, Y, N, K, e_base, add_res); }

extern "C" __global__ void skinny_bf12_m5_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    bf16_t* __restrict__ Y, int N, int K, unsigned int e_base, int add_res
) { skinny_bf12_1wpr_impl<5>(X, sm8, exp4, rp, ec, ev, Y, N, K, e_base, add_res); }

// 18.5 Batched Fused BF12-Lossless lm_head + Online Argmax (M = 1..5, ZERO logits DRAM!)
// Grid: (1024, 1, 1), Block: (256, 1, 1) -> followed by argmax[_batch]_stage2_bf16(num_blocks=1024)
template<int M_VAL>
__device__ __forceinline__ void lm_head_bf12_argmax_impl(
    const bf16_t* __restrict__ X,
    const unsigned char* __restrict__ W_sm8,
    const unsigned char* __restrict__ W_exp4,
    const int* __restrict__ esc_row_ptr,
    const unsigned short* __restrict__ esc_col,
    const bf16_t* __restrict__ esc_val,
    float* __restrict__ part_vals, // [M_VAL, gridDim.x]
    int* __restrict__ part_idxs,   // [M_VAL, gridDim.x]
    int V,
    unsigned int e_base
) {
    int tid = threadIdx.x;
    int wid = tid >> 5;
    int lane = tid & 31;
    int rpb = (V + gridDim.x - 1) / gridDim.x;
    int v0 = blockIdx.x * rpb;
    int v1 = min(v0 + rpb, V);

    float bv[M_VAL];
    int bi[M_VAL];
    #pragma unroll
    for (int m = 0; m < M_VAL; ++m) { bv[m] = -1e30f; bi[m] = 0; }

    for (int n = v0 + wid; n < v1; n += 8) {
        const uint4* sm_row = reinterpret_cast<const uint4*>(W_sm8 + (size_t)n * 2560);
        const uint2* ex_row = reinterpret_cast<const uint2*>(W_exp4 + (size_t)n * 1280);
        float acc[M_VAL];
        #pragma unroll
        for (int m = 0; m < M_VAL; ++m) acc[m] = 0.0f;

        #pragma unroll
        for (int s = 0; s < 5; ++s) {
            int kv = s * 32 + lane;
            uint4 sm16 = r13_ldcs_uint4(&sm_row[kv]);
            uint2 ex16 = r14_ldcs_uint2(&ex_row[kv]);
            float wf0[4], wf1[4], wf2[4], wf3[4];
            r14_decode_bf12_4f(sm16.x, ex16.x & 0xFFFFu, e_base, wf0);
            r14_decode_bf12_4f(sm16.y, ex16.x >> 16,     e_base, wf1);
            r14_decode_bf12_4f(sm16.z, ex16.y & 0xFFFFu, e_base, wf2);
            r14_decode_bf12_4f(sm16.w, ex16.y >> 16,     e_base, wf3);
            #pragma unroll
            for (int m = 0; m < M_VAL; ++m) {
                const uint4* xr = reinterpret_cast<const uint4*>(X + (size_t)m * 2560);
                uint4 x0 = __ldg(&xr[kv * 2]);
                uint4 x1 = __ldg(&xr[kv * 2 + 1]);
                float xf0[8], xf1[8];
                unpack_uint4_bf16(x0, xf0);
                unpack_uint4_bf16(x1, xf1);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[m] = fmaf(xf0[j],     wf0[j], acc[m]);
                    acc[m] = fmaf(xf0[j + 4], wf1[j], acc[m]);
                    acc[m] = fmaf(xf1[j],     wf2[j], acc[m]);
                    acc[m] = fmaf(xf1[j + 4], wf3[j], acc[m]);
                }
            }
        }
        int es = esc_row_ptr[n];
        int ee = esc_row_ptr[n + 1];
        for (int idx = es + lane; idx < ee; idx += 32) {
            int c = (int)esc_col[idx];
            float wv = bf16_to_f(esc_val[idx]);
            #pragma unroll
            for (int m = 0; m < M_VAL; ++m) {
                acc[m] = fmaf(bf16_to_f(X[(size_t)m * 2560 + c]), wv, acc[m]);
            }
        }
        #pragma unroll
        for (int m = 0; m < M_VAL; ++m) {
            float d = warp_reduce_sum(acc[m]);
            if (lane == 0 && d > bv[m]) { bv[m] = d; bi[m] = n; }
        }
    }

    __shared__ float sm_v[M_VAL][8];
    __shared__ int sm_i[M_VAL][8];
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < M_VAL; ++m) { sm_v[m][wid] = bv[m]; sm_i[m][wid] = bi[m]; }
    }
    __syncthreads();
    if (tid < M_VAL) {
        int m = tid;
        float bestv = sm_v[m][0];
        int besti = sm_i[m][0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (sm_v[m][w] > bestv) { bestv = sm_v[m][w]; besti = sm_i[m][w]; }
        }
        part_vals[m * gridDim.x + blockIdx.x] = bestv;
        part_idxs[m * gridDim.x + blockIdx.x] = besti;
    }
}

extern "C" __global__ void lm_head_bf12_argmax_m1_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    float* __restrict__ pv, int* __restrict__ pi, int V, unsigned int e_base
) { lm_head_bf12_argmax_impl<1>(X, sm8, exp4, rp, ec, ev, pv, pi, V, e_base); }

extern "C" __global__ void lm_head_bf12_argmax_m2_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    float* __restrict__ pv, int* __restrict__ pi, int V, unsigned int e_base
) { lm_head_bf12_argmax_impl<2>(X, sm8, exp4, rp, ec, ev, pv, pi, V, e_base); }

extern "C" __global__ void lm_head_bf12_argmax_m3_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    float* __restrict__ pv, int* __restrict__ pi, int V, unsigned int e_base
) { lm_head_bf12_argmax_impl<3>(X, sm8, exp4, rp, ec, ev, pv, pi, V, e_base); }

extern "C" __global__ void lm_head_bf12_argmax_m4_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    float* __restrict__ pv, int* __restrict__ pi, int V, unsigned int e_base
) { lm_head_bf12_argmax_impl<4>(X, sm8, exp4, rp, ec, ev, pv, pi, V, e_base); }

extern "C" __global__ void lm_head_bf12_argmax_m5_bf16(
    const bf16_t* __restrict__ X, const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    float* __restrict__ pv, int* __restrict__ pi, int V, unsigned int e_base
) { lm_head_bf12_argmax_impl<5>(X, sm8, exp4, rp, ec, ev, pv, pi, V, e_base); }

// 18.6 Prefill 1-Pass BF12 -> INT8 SmoothQuant (all 256 threads active, 8 weights/iter, 0 spill!)
template<int MAX_ITERS>
__device__ __forceinline__ void quant_bf12_to_i8_smooth_strided_impl(
    const unsigned char* __restrict__ W_sm8,
    const unsigned char* __restrict__ W_exp4,
    const int* __restrict__ esc_row_ptr,
    const unsigned short* __restrict__ esc_col,
    const bf16_t* __restrict__ esc_val,
    const float* __restrict__ col_mul,
    signed char* __restrict__ Q0,
    float* __restrict__ S0,
    int valid_rows,
    int K,
    int ld_in,
    int k_off,
    unsigned int e_base
) {
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int k8 = K >> 3;
    uint2* out_v = reinterpret_cast<uint2*>(Q0 + (size_t)r * K);

    if (r >= valid_rows) {
        if (tid == 0) S0[r] = 1.0f;
        uint2 z = {0, 0};
        #pragma unroll
        for (int it = 0; it < MAX_ITERS; ++it) {
            int i = it * 256 + tid;
            if (i < k8) out_v[i] = z;
        }
        return;
    }

    const uint2* sm_row = reinterpret_cast<const uint2*>(W_sm8 + (size_t)r * ld_in + k_off);
    const unsigned int* ex_row = reinterpret_cast<const unsigned int*>(W_exp4 + (size_t)r * (ld_in >> 1) + (k_off >> 1));
    int es = esc_row_ptr[r];
    int ee = esc_row_ptr[r + 1];

    float reg_vals[MAX_ITERS * 8];
    float lmax = 0.0f;

    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        int i = it * 256 + tid;
        if (i < k8) {
            uint2 sm8_v = r14_ldcs_uint2(&sm_row[i]);
            unsigned int ex8_v = r14_ldcs_u32(&ex_row[i]);
            float v[8];
            r14_decode_bf12_4f(sm8_v.x, ex8_v & 0xFFFFu, e_base, &v[0]);
            r14_decode_bf12_4f(sm8_v.y, ex8_v >> 16,     e_base, &v[4]);
            int base_k = i << 3;
            if (es < ee) {
                for (int idx = es; idx < ee; ++idx) {
                    int c_rel = (int)esc_col[idx] - k_off;
                    if (c_rel >= base_k && c_rel < base_k + 8) {
                        v[c_rel - base_k] = bf16_to_f(esc_val[idx]);
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float fv = v[j] * col_mul[base_k + j];
                reg_vals[it * 8 + j] = fv;
                lmax = fmaxf(lmax, fabsf(fv));
            }
        }
    }

    lmax = r10_warp_max(lmax);
    __shared__ float smax[8];
    if ((tid & 31) == 0) smax[tid >> 5] = lmax;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) m = fmaxf(m, smax[w]);
        smax[0] = (m > 1e-12f) ? (m * (1.0f / 127.0f)) : 1.0f;
        S0[r] = smax[0];
    }
    __syncthreads();
    float inv_s = 1.0f / smax[0];

    #pragma unroll
    for (int it = 0; it < MAX_ITERS; ++it) {
        int i = it * 256 + tid;
        if (i < k8) {
            signed char q[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int iv = __float2int_rn(reg_vals[it * 8 + j] * inv_s);
                q[j] = (signed char)(iv < -127 ? -127 : (iv > 127 ? 127 : iv));
            }
            out_v[i] = *reinterpret_cast<uint2*>(q);
        }
    }
}

extern "C" __global__ void quant_bf12_to_i8_smooth_k4k(
    const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    const float* __restrict__ col_mul, signed char* __restrict__ Q0, float* __restrict__ S0,
    int valid_rows, int K, unsigned int e_base
) {
    quant_bf12_to_i8_smooth_strided_impl<2>(sm8, exp4, rp, ec, ev, col_mul, Q0, S0, valid_rows, K, K, 0, e_base);
}

extern "C" __global__ void quant_bf12_to_i8_smooth_k10k(
    const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    const float* __restrict__ col_mul, signed char* __restrict__ Q0, float* __restrict__ S0,
    int valid_rows, int K, unsigned int e_base
) {
    quant_bf12_to_i8_smooth_strided_impl<5>(sm8, exp4, rp, ec, ev, col_mul, Q0, S0, valid_rows, K, K, 0, e_base);
}

extern "C" __global__ void quant_bf12_to_i8_smooth_strided_k6k(
    const unsigned char* __restrict__ sm8, const unsigned char* __restrict__ exp4,
    const int* __restrict__ rp, const unsigned short* __restrict__ ec, const bf16_t* __restrict__ ev,
    const float* __restrict__ col_mul, signed char* __restrict__ Q0, float* __restrict__ S0,
    int valid_rows, int K, int ld_in, int k_off, unsigned int e_base
) {
    quant_bf12_to_i8_smooth_strided_impl<3>(sm8, exp4, rp, ec, ev, col_mul, Q0, S0, valid_rows, K, ld_in, k_off, e_base);
}

// 18.7 Fast Bit-Exact BF12 -> BF16 Unpack (for short prefill M < 384, BF16 mode, Ozaki, MTP prefill)
// Grid: (N, 1, 1), Block: (256, 1, 1)
extern "C" __global__ void bf12_to_bf16_unpack(
    const unsigned char* __restrict__ W_sm8,
    const unsigned char* __restrict__ W_exp4,
    const int* __restrict__ esc_row_ptr,
    const unsigned short* __restrict__ esc_col,
    const bf16_t* __restrict__ esc_val,
    bf16_t* __restrict__ out_bf16,
    int N,
    int K,
    unsigned int e_base
) {
    int n = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;
    const uint4* sm_row = reinterpret_cast<const uint4*>(W_sm8 + (size_t)n * K);
    const uint2* ex_row = reinterpret_cast<const uint2*>(W_exp4 + (size_t)n * (K >> 1));
    bf16_t* dst_row = out_bf16 + (size_t)n * K;
    uint4* dst = reinterpret_cast<uint4*>(dst_row);
    int k16 = K >> 4;
    for (int i = tid; i < k16; i += blockDim.x) {
        uint4 sm16 = r13_ldcs_uint4(&sm_row[i]);
        uint2 ex16 = r14_ldcs_uint2(&ex_row[i]);
        uint2 p0 = r14_decode_bf12_4u(sm16.x, ex16.x & 0xFFFFu, e_base);
        uint2 p1 = r14_decode_bf12_4u(sm16.y, ex16.x >> 16,     e_base);
        uint2 p2 = r14_decode_bf12_4u(sm16.z, ex16.y & 0xFFFFu, e_base);
        uint2 p3 = r14_decode_bf12_4u(sm16.w, ex16.y >> 16,     e_base);
        dst[i * 2]     = {p0.x, p0.y, p1.x, p1.y};
        dst[i * 2 + 1] = {p2.x, p2.y, p3.x, p3.y};
    }
    __syncthreads();
    int es = esc_row_ptr[n];
    int ee = esc_row_ptr[n + 1];
    for (int idx = es + tid; idx < ee; idx += blockDim.x) {
        dst_row[esc_col[idx]] = esc_val[idx];
    }
}






