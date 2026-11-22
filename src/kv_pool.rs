//! Phase R7: Paged KV Block Manager (BLOCK_SIZE = 16) + Multi-Slot Gated DeltaNet Pool.
//! Zero-fragmentation single-arena GPU pool for Continuous Batching (B = 1..4) on RTX 3080 10 GB.

use anyhow::Result;
use crate::cuda_ffi::{CUdeviceptr, CudaRuntime};

pub const KV_BLOCK_SIZE: usize = 16;
pub const MAX_BLOCKS_PER_SEQ: usize = 256; // supports up to 256 * 16 = 4,096 tokens/sequence
pub const KV_BLOCK_BYTES: usize = 8 * 2 * 4 * KV_BLOCK_SIZE * 256 * 2; // 512 KiB per block (8 Full-Attn layers!)

pub struct KvBlockPool {
    pub arena: CUdeviceptr,
    pub bytes: usize,
    pub total_blocks: usize,
    free_stack: Vec<i32>,
}

impl KvBlockPool {
    pub fn alloc(cuda: &CudaRuntime, num_blocks: usize) -> Result<Self> {
        let bytes = num_blocks * KV_BLOCK_BYTES;
        let arena = cuda.alloc(bytes)?;
        cuda.memset_zero_async(arena, bytes)?;
        cuda.sync()?;
        let free_stack: Vec<i32> = (0..num_blocks as i32).rev().collect();
        Ok(Self {
            arena,
            bytes,
            total_blocks: num_blocks,
            free_stack,
        })
    }

    pub fn free_count(&self) -> usize {
        self.free_stack.len()
    }

    pub fn used_count(&self) -> usize {
        self.total_blocks - self.free_stack.len()
    }

    pub fn blocks_needed(tokens: usize) -> usize {
        (tokens + KV_BLOCK_SIZE - 1) / KV_BLOCK_SIZE
    }

    pub fn alloc_blocks(&mut self, n: usize) -> Option<Vec<i32>> {
        if self.free_stack.len() < n {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.free_stack.pop().unwrap());
        }
        Some(out)
    }

    pub fn alloc_one(&mut self) -> Option<i32> {
        self.free_stack.pop()
    }

    pub fn free_blocks(&mut self, blocks: &[i32]) {
        self.free_stack.extend_from_slice(blocks);
    }
}

pub struct DeltaNetSlotPool {
    pub conv_arena: CUdeviceptr, // [max_batch, 24, 8192, 4] bf16 (1.5 MiB / slot)
    pub rec_arena: CUdeviceptr,  // [max_batch, 24, 32, 128, 128] f32 (48.0 MiB / slot)
    pub bytes: usize,
    pub max_batch: usize,
    free_slots: Vec<usize>,
}

impl DeltaNetSlotPool {
    pub fn alloc(cuda: &CudaRuntime, max_batch: usize) -> Result<Self> {
        let conv_bytes = max_batch * 24 * 8192 * 4 * 2;
        let rec_bytes = max_batch * 24 * 32 * 128 * 128 * 4;
        let conv_arena = cuda.alloc(conv_bytes)?;
        let rec_arena = cuda.alloc(rec_bytes)?;
        cuda.memset_zero_async(conv_arena, conv_bytes)?;
        cuda.memset_zero_async(rec_arena, rec_bytes)?;
        cuda.sync()?;
        let free_slots: Vec<usize> = (0..max_batch).rev().collect();
        Ok(Self {
            conv_arena,
            rec_arena,
            bytes: conv_bytes + rec_bytes,
            max_batch,
            free_slots,
        })
    }

    pub fn acquire(&mut self) -> Option<usize> {
        self.free_slots.pop()
    }

    pub fn release(&mut self, slot: usize) {
        self.free_slots.push(slot);
    }

    pub fn conv_ptr(&self, slot: usize, lin_idx: usize) -> CUdeviceptr {
        self.conv_arena + (((slot * 24 + lin_idx) * 32768) * 2) as u64
    }

    pub fn rec_ptr(&self, slot: usize, lin_idx: usize) -> CUdeviceptr {
        self.rec_arena + (((slot * 24 + lin_idx) * 524288) * 4) as u64
    }
}
