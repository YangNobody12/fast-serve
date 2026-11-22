# fast-serve (`qwen35_split_server`)

A **Pure Rust + Custom CUDA / WMMA Tensor Core** 2-Part Split Inference Serving Engine for **Qwen3.5-4B (Pure BF16, 0% model modification)** — zero Python, zero PyTorch, direct dynamic FFI to `nvcuda.dll` + `cublas64_12.dll`, and `axum 0.8` OpenAI-compatible SSE streaming.

## Key Engineering Highlights

- **2-Part Split Architecture (`L0..15` → Zero-Copy GPU Bridge → `L16..31`)** with single-arena VRAM allocation (`0 bytes register spill` across all **48 custom CUDA kernels** on `sm_86`).
- **8-Warp `uint4` Coalesced Skinny BF16 GEMM (`M = 1..4`) + Fused Residual Add**: achieves **`92.0–92.8%` of RTX 3080 GDDR6X Peak (`680–686 GB/s`)** verified via Nsight Compute (`ncu 2025.1`).
- **Native Qwen3.5 MTP Speculative Decoding (`K = 3`) with 2-Graph Full-Cycle CUDA Graphs**: collapses an entire `~3.14`-token cycle (Commit DeltaNet + Draft `d1..d3` + Parallel Verify `M = 4` + D2H) into **2 CUDA Graph launches** (`0` eager kernels, **`98.45%` True GPU Busy**).
- **FlashLinearAttention-Style Chunked WY Gated DeltaNet Prefill (WMMA BF16 Tensor Cores)**: single-kernel `nvcuda::wmma` FP32-register state recurrence + `2×32` warp-shuffle block triangular inversion + `batch=128` BF16 batched GEMMs (`+0.00 MiB` extra VRAM).
- **2D Parallel Causal Conv1D (`512` blocks) + WMMA BF16 Tensor Core FlashAttention-2** (Contiguous & Paged KV).
- **3-Tier Hybrid Prefix Cache (`KV + DeltaNet + MTP`)**: Tier 0 Live GPU Hot-Cache + Tier 1 Recycled Host-RAM Pool via `16 MiB` Batched Pinned Staging + single-pass in-flight `snap_t` snapshots (`+0.00 MiB` extra VRAM).
- **Continuous Batching (`--max-batch 4`) + Paged KV Block Manager (`BLOCK_SIZE = 16`)**: pre-captured Batched Decode CUDA Graphs (`B = 1..4`) + single contiguous pinned metadata transfer + in-graph D2H (**`98.06%` True GPU Busy**, `4/4` Bit-Exact isolation, `0` KV block leaks).
- **Auto Windows WDDM VRAM Headroom Guard (`450 MiB`)**: automatically clamps `max_seq` when desktop apps consume VRAM to prevent silent PCIe paging drops (`--force-max-seq` to override).

## Verified Benchmarks (RTX 3080 10 GB · `760 GB/s` · Pure BF16 Untouched)

| Workload / Benchmark | Result | Hardware Metrics (`nsys 10 kHz` / `ncu 2025.1`) |
|---|---|---|
| **Single-User Decode (`MTP K=3`, `--test-only`)** | **`136–158.6 tok/s`** (`6.31 ms/tok`) | `94.34%` SMs Active · `674.2 GB/s` DRAM Read (`88.7%` / `99%` p95) |
| **Single-User `DECODE_64` (`nsys 10 kHz`)** | **`461.8 ms`** (`7.22 ms/tok` · `138.6 tok/s`) | `98.45%` True GPU Busy · `12.4 GB/s` DRAM Write |
| **AIPerf `c = 1` (`128 in → 64 out`, `0%` hit)** | **`111.45 tok/s/user`** (`max 140.06`, `ITL 9.08 ms`) | `Goodput 1.61 req/s` (`20/20 PASS`, `0%` err) |
| **Prefill (`PREFILL_512` / `PREFILL_128`)** | **`91.75 ms` (`5,580 tok/s`)** / **`29.56 ms`** | `29.0%` Active Tensor Core · `5.77×` faster 2D Conv1D |
| **Continuous Batching (`--max-batch 4`, `nsys`)** | **`1,048.2 ms`** (`244.2 tok/s` for 64×4 tok) | **`98.06%` True GPU Busy** · `4/4` Bit-Exact · `0` KV leaks |
| **AIPerf `c = 4` (`128 in → 64 out`, `24/24`)** | **`225.6–227.4 tok/s`** (`ITL 16.0–16.2 ms`) | `Goodput 3.52–3.55 req/s` (`100%` ≤ 20 ms SLO) |

## Repository Layout

- `kernels/qwen35.cu` — 48 zero-spill custom CUDA / WMMA BF16 kernels compiled to `sm_86` PTX via `build.rs`.
- `src/cuda_ffi.rs` — Direct dynamic FFI bindings to `nvcuda.dll` & `cublas64_12.dll` (`cublasGemmEx`, `cublasGemmStridedBatchedEx`, CUDA Graphs).
- `src/engine.rs` — Single-user 2-Part Split Engine (2-Graph MTP `K=3`, 3-Tier Prefix Cache, WMMA Chunked WY DeltaNet, WDDM Guard).
- `src/batch_engine.rs` — Iteration-level Continuous Batching Engine (`B = 1..4`) with Batched Decode CUDA Graphs & Paged KV.
- `src/kv_pool.rs` — `O(1)` Paged KV Block Manager (`BLOCK_SIZE = 16`) & DeltaNet Slot Pool.
- `src/loader.rs` — Zero-copy `safetensors` Pure BF16 split weight packer.
- `src/api.rs` — `axum 0.8` OpenAI `/v1/chat/completions` SSE server (`TCP_NODELAY`) + `/api/fast_status`.
- `src/tokenizer.rs` / `src/config.rs` / `src/main.rs` — Qwen3.5 tokenizer, config, and CLI entrypoint.

## Build & Run

Requires Rust toolchain (`cargo`) and NVIDIA CUDA Toolkit (`nvcc` targeting `sm_86` in `build.rs`).

```powershell
cargo build --release

# Single-user ultra-fast mode (MTP K=3 + 3-Tier Prefix Cache)
.\target\release\qwen35_split_server.exe --model D:\serving\Qwen3.5-4B --port 8000 --mtp-drafts 3 --prefix-cache-slots 8

# Multi-user Continuous Batching + Paged KV mode (B = 1..4)
.\target\release\qwen35_split_server.exe --model D:\serving\Qwen3.5-4B --port 8000 --max-batch 4

# Quick self-test
.\target\release\qwen35_split_server.exe --model D:\serving\Qwen3.5-4B --test-only --mtp-drafts 3
```
