//! Zero-copy `mmap` safetensors loader with Single-Arena GPU packing.
//! Eliminates PyTorch/CuPy VRAM fragmentation (~350-500 MiB saved!) and keeps Host RAM < 200 MB.

use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::{self, File};
use std::path::Path;

use crate::config::TextConfig;
use crate::cuda_ffi::{CUdeviceptr, CudaRuntime};

#[derive(Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Clone, Copy)]
pub struct PackedLayerPtrs {
    pub is_linear: bool,
    pub in_norm: CUdeviceptr,
    pub post_norm: CUdeviceptr,
    pub gate_up: CUdeviceptr, // [18432, 2560] bf16
    pub down: CUdeviceptr,    // [2560, 9216] bf16
    // DeltaNet
    pub in_all: CUdeviceptr,  // [12352, 2560] bf16
    pub out_proj: CUdeviceptr,// [2560, 4096] bf16
    pub conv_w: CUdeviceptr,  // [8192, 4] bf16
    pub dt_bias: CUdeviceptr, // [32] bf16
    pub a_log_f32: CUdeviceptr,// [32] f32
    pub dn_norm: CUdeviceptr, // [128] bf16
    // Full Attention
    pub qkv: CUdeviceptr,     // [10240, 2560] bf16
    pub o_proj: CUdeviceptr,  // [2560, 4096] bf16
    pub q_norm: CUdeviceptr,  // [256] bf16
    pub k_norm: CUdeviceptr,  // [256] bf16
}

#[derive(Clone, Copy)]
pub struct MtpWeights {
    pub arena: CUdeviceptr,
    pub bytes: usize,
    pub pre_fc_norm_emb: CUdeviceptr, // [2560] bf16
    pub pre_fc_norm_hid: CUdeviceptr, // [2560] bf16
    pub fc_w: CUdeviceptr,            // [2560, 5120] bf16
    pub layer: PackedLayerPtrs,       // Full Attention layer mtp.layers.0
    pub norm_w: CUdeviceptr,          // [2560] bf16
}

pub struct LoadedSplitWeights {
    pub p1_arena: CUdeviceptr,
    pub p1_bytes: usize,
    pub p2_arena: CUdeviceptr,
    pub p2_bytes: usize,
    pub embed_w: CUdeviceptr,
    pub lm_head_w: CUdeviceptr, // Tied == embed_w!
    pub final_norm_w: CUdeviceptr,
    pub p1_layers: Vec<PackedLayerPtrs>,
    pub p2_layers: Vec<PackedLayerPtrs>,
    pub mtp: Option<MtpWeights>,
}

struct ArenaBump {
    base: CUdeviceptr,
    offset: usize,
    capacity: usize,
}

impl ArenaBump {
    fn new(base: CUdeviceptr, capacity: usize) -> Self {
        Self { base, offset: 0, capacity }
    }
    fn alloc_aligned(&mut self, bytes: usize) -> Result<CUdeviceptr> {
        let align = 256usize;
        let off = (self.offset + align - 1) & !(align - 1);
        if off + bytes > self.capacity {
            return Err(anyhow!("Arena overflow: need {} > cap {}", off + bytes, self.capacity));
        }
        self.offset = off + bytes;
        Ok(self.base + off as u64)
    }
}

fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn f32_to_bf16(v: f32) -> u16 {
    let mut u = v.to_bits();
    let lsb = (u >> 16) & 1;
    u = u.wrapping_add(0x7FFF + lsb);
    (u >> 16) as u16
}

struct ShardStore {
    weight_map: HashMap<String, String>,
    mmaps: HashMap<String, Mmap>,
}

impl ShardStore {
    fn open(model_dir: &Path) -> Result<Self> {
        let idx_str = fs::read_to_string(model_dir.join("model.safetensors.index.json"))?;
        let idx: SafetensorsIndex = serde_json::from_str(&idx_str)?;
        let mut mmaps = HashMap::new();
        for fname in idx.weight_map.values() {
            if !mmaps.contains_key(fname) {
                let file = File::open(model_dir.join(fname))
                    .with_context(|| format!("open shard {fname}"))?;
                let mmap = unsafe { Mmap::map(&file)? };
                mmaps.insert(fname.clone(), mmap);
            }
        }
        Ok(Self { weight_map: idx.weight_map, mmaps })
    }

    fn parse_all(&self) -> Result<HashMap<&str, SafeTensors<'_>>> {
        let mut out = HashMap::new();
        for (fname, mm) in &self.mmaps {
            out.insert(fname.as_str(), SafeTensors::deserialize(&mm[..])?);
        }
        Ok(out)
    }
}

fn get_bf16_bytes<'a>(
    key: &str,
    wm: &HashMap<String, String>,
    shards: &'a HashMap<&str, SafeTensors<'a>>,
    scratch: &mut Vec<u8>,
) -> Result<&'a [u8]> {
    let fname = wm.get(key).ok_or_else(|| anyhow!("Missing key: {key}"))?;
    let st = shards.get(fname.as_str()).unwrap();
    let tv = st.tensor(key)?;
    match tv.dtype() {
        Dtype::BF16 => Ok(tv.data()),
        Dtype::F32 => {
            // Convert f32 -> bf16 into scratch if needed
            let raw = tv.data();
            let n = raw.len() / 4;
            scratch.clear();
            scratch.reserve(n * 2);
            for i in 0..n {
                let f = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
                let b = f32_to_bf16(f);
                scratch.extend_from_slice(&b.to_le_bytes());
            }
            // Note: caller uses helper below when conversion needed
            Err(anyhow!("USE_CONVERT"))
        }
        dt => Err(anyhow!("Unsupported dtype {dt:?} for {key}")),
    }
}

fn upload_bf16(
    cuda: &CudaRuntime,
    arena: &mut ArenaBump,
    key: &str,
    wm: &HashMap<String, String>,
    shards: &HashMap<&str, SafeTensors<'_>>,
) -> Result<CUdeviceptr> {
    let fname = wm.get(key).ok_or_else(|| anyhow!("Missing key: {key}"))?;
    let tv = shards.get(fname.as_str()).unwrap().tensor(key)?;
    let mut conv = Vec::new();
    let bytes: &[u8] = match tv.dtype() {
        Dtype::BF16 => tv.data(),
        Dtype::F32 => {
            let raw = tv.data();
            let n = raw.len() / 4;
            conv.reserve(n * 2);
            for i in 0..n {
                let f = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
                conv.extend_from_slice(&f32_to_bf16(f).to_le_bytes());
            }
            &conv
        }
        dt => return Err(anyhow!("Unexpected dtype {dt:?} for {key}")),
    };
    let ptr = arena.alloc_aligned(bytes.len())?;
    cuda.htod_sync(ptr, bytes.as_ptr(), bytes.len())?;
    Ok(ptr)
}

fn upload_f32(
    cuda: &CudaRuntime,
    arena: &mut ArenaBump,
    key: &str,
    wm: &HashMap<String, String>,
    shards: &HashMap<&str, SafeTensors<'_>>,
) -> Result<CUdeviceptr> {
    let fname = wm.get(key).ok_or_else(|| anyhow!("Missing key: {key}"))?;
    let tv = shards.get(fname.as_str()).unwrap().tensor(key)?;
    let mut conv = Vec::new();
    let bytes: &[u8] = match tv.dtype() {
        Dtype::F32 => tv.data(),
        Dtype::BF16 => {
            let raw = tv.data();
            let n = raw.len() / 2;
            conv.reserve(n * 4);
            for i in 0..n {
                let b = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                conv.extend_from_slice(&bf16_to_f32(b).to_le_bytes());
            }
            &conv
        }
        dt => return Err(anyhow!("Unexpected dtype {dt:?} for {key}")),
    };
    let ptr = arena.alloc_aligned(bytes.len())?;
    cuda.htod_sync(ptr, bytes.as_ptr(), bytes.len())?;
    Ok(ptr)
}

fn upload_packed_bf16(
    cuda: &CudaRuntime,
    arena: &mut ArenaBump,
    keys: &[String],
    wm: &HashMap<String, String>,
    shards: &HashMap<&str, SafeTensors<'_>>,
    staging: &mut Vec<u8>,
) -> Result<CUdeviceptr> {
    staging.clear();
    let mut total = 0usize;
    for k in keys {
        let fname = wm.get(k).ok_or_else(|| anyhow!("Missing key: {k}"))?;
        let tv = shards.get(fname.as_str()).unwrap().tensor(k)?;
        total += match tv.dtype() {
            Dtype::BF16 => tv.data().len(),
            Dtype::F32 => tv.data().len() / 2,
            dt => return Err(anyhow!("Unexpected dtype {dt:?} for {k}")),
        };
    }
    staging.reserve(total);
    let mut dummy = Vec::new();
    for k in keys {
        match get_bf16_bytes(k, wm, shards, &mut dummy) {
            Ok(slice) => staging.extend_from_slice(slice),
            Err(_) => {
                let fname = wm.get(k).unwrap();
                let raw = shards.get(fname.as_str()).unwrap().tensor(k)?.data();
                let n = raw.len() / 4;
                for i in 0..n {
                    let f = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
                    staging.extend_from_slice(&f32_to_bf16(f).to_le_bytes());
                }
            }
        }
    }
    let ptr = arena.alloc_aligned(staging.len())?;
    cuda.htod_sync(ptr, staging.as_ptr(), staging.len())?;
    Ok(ptr)
}

fn calc_layer_bytes(is_linear: bool, cfg: &TextConfig) -> usize {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let align = |b: usize| (b + 255) & !255;
    let mut sum = 0usize;
    sum += align(h * 2) * 2;               // in_norm + post_norm
    sum += align(2 * inter * h * 2);       // gate_up
    sum += align(h * inter * 2);           // down
    if is_linear {
        sum += align(12352 * h * 2);       // in_all
        sum += align(h * 4096 * 2);        // out_proj
        sum += align(8192 * 4 * 2);        // conv_w
        sum += align(32 * 2);              // dt_bias
        sum += align(32 * 4);              // a_log_f32
        sum += align(128 * 2);             // dn_norm
    } else {
        sum += align(10240 * h * 2);       // qkv
        sum += align(h * 4096 * 2);        // o_proj
        sum += align(256 * 2) * 2;         // q_norm + k_norm
    }
    sum
}

pub fn load_split_weights(
    cuda: &CudaRuntime,
    model_dir: &Path,
    cfg: &TextConfig,
    split_layer: usize,
    load_mtp: bool,
) -> Result<LoadedSplitWeights> {
    let store = ShardStore::open(model_dir)?;
    let shards = store.parse_all()?;
    let align = |b: usize| (b + 255) & !255;

    // Exact arena sizes for Part 1 (embed + layers 0..split) and Part 2 (layers split..32 + final_norm)
    let mut p1_cap = align(cfg.vocab_size * cfg.hidden_size * 2) + 4096;
    for i in 0..split_layer {
        p1_cap += calc_layer_bytes(cfg.layer_types[i] == "linear_attention", cfg);
    }
    let mut p2_cap = align(cfg.hidden_size * 2) + 4096;
    for i in split_layer..cfg.num_hidden_layers {
        p2_cap += calc_layer_bytes(cfg.layer_types[i] == "linear_attention", cfg);
    }

    let p1_arena = cuda.alloc(p1_cap)?;
    let p2_arena = cuda.alloc(p2_cap)?;
    let mut b1 = ArenaBump::new(p1_arena, p1_cap);
    let mut b2 = ArenaBump::new(p2_arena, p2_cap);

    let embed_w = upload_bf16(cuda, &mut b1, "model.language_model.embed_tokens.weight", &store.weight_map, &shards)?;
    let lm_head_w = embed_w; // Tied embeddings (saves 1,212 MiB!)
    let final_norm_w = upload_bf16(cuda, &mut b2, "model.language_model.norm.weight", &store.weight_map, &shards)?;

    let mut staging: Vec<u8> = Vec::with_capacity(18432 * 2560 * 2);
    let mut p1_layers = Vec::with_capacity(split_layer);
    let mut p2_layers = Vec::with_capacity(cfg.num_hidden_layers - split_layer);

    for i in 0..cfg.num_hidden_layers {
        let is_linear = cfg.layer_types[i] == "linear_attention";
        let arena = if i < split_layer { &mut b1 } else { &mut b2 };
        let pfx = format!("model.language_model.layers.{i}");

        let in_norm = upload_bf16(cuda, arena, &format!("{pfx}.input_layernorm.weight"), &store.weight_map, &shards)?;
        let post_norm = upload_bf16(cuda, arena, &format!("{pfx}.post_attention_layernorm.weight"), &store.weight_map, &shards)?;
        let gate_up = upload_packed_bf16(
            cuda,
            arena,
            &[format!("{pfx}.mlp.gate_proj.weight"), format!("{pfx}.mlp.up_proj.weight")],
            &store.weight_map,
            &shards,
            &mut staging,
        )?;
        let down = upload_bf16(cuda, arena, &format!("{pfx}.mlp.down_proj.weight"), &store.weight_map, &shards)?;

        let lyr = if is_linear {
            let in_all = upload_packed_bf16(
                cuda,
                arena,
                &[
                    format!("{pfx}.linear_attn.in_proj_qkv.weight"),
                    format!("{pfx}.linear_attn.in_proj_z.weight"),
                    format!("{pfx}.linear_attn.in_proj_b.weight"),
                    format!("{pfx}.linear_attn.in_proj_a.weight"),
                ],
                &store.weight_map,
                &shards,
                &mut staging,
            )?;
            let out_proj = upload_bf16(cuda, arena, &format!("{pfx}.linear_attn.out_proj.weight"), &store.weight_map, &shards)?;
            let conv_w = upload_bf16(cuda, arena, &format!("{pfx}.linear_attn.conv1d.weight"), &store.weight_map, &shards)?;
            let dt_bias = upload_bf16(cuda, arena, &format!("{pfx}.linear_attn.dt_bias"), &store.weight_map, &shards)?;
            let a_log_f32 = upload_f32(cuda, arena, &format!("{pfx}.linear_attn.A_log"), &store.weight_map, &shards)?;
            let dn_norm = upload_bf16(cuda, arena, &format!("{pfx}.linear_attn.norm.weight"), &store.weight_map, &shards)?;
            PackedLayerPtrs {
                is_linear: true,
                in_norm, post_norm, gate_up, down,
                in_all, out_proj, conv_w, dt_bias, a_log_f32, dn_norm,
                qkv: 0, o_proj: 0, q_norm: 0, k_norm: 0,
            }
        } else {
            let qkv = upload_packed_bf16(
                cuda,
                arena,
                &[
                    format!("{pfx}.self_attn.q_proj.weight"),
                    format!("{pfx}.self_attn.k_proj.weight"),
                    format!("{pfx}.self_attn.v_proj.weight"),
                ],
                &store.weight_map,
                &shards,
                &mut staging,
            )?;
            let o_proj = upload_bf16(cuda, arena, &format!("{pfx}.self_attn.o_proj.weight"), &store.weight_map, &shards)?;
            let q_norm = upload_bf16(cuda, arena, &format!("{pfx}.self_attn.q_norm.weight"), &store.weight_map, &shards)?;
            let k_norm = upload_bf16(cuda, arena, &format!("{pfx}.self_attn.k_norm.weight"), &store.weight_map, &shards)?;
            PackedLayerPtrs {
                is_linear: false,
                in_norm, post_norm, gate_up, down,
                in_all: 0, out_proj: 0, conv_w: 0, dt_bias: 0, a_log_f32: 0, dn_norm: 0,
                qkv, o_proj, q_norm, k_norm,
            }
        };

        if i < split_layer {
            p1_layers.push(lyr);
        } else {
            p2_layers.push(lyr);
        }
    }

    // Optional built-in Qwen3.5 MTP Speculative Draft Head (~230 MiB)
    let mtp = if load_mtp && store.weight_map.contains_key("mtp.fc.weight") {
        let h = cfg.hidden_size;
        let mtp_cap = align(h * 2) * 3 + align(h * 2 * h * 2) + calc_layer_bytes(false, cfg) + 4096;
        let mtp_arena = cuda.alloc(mtp_cap)?;
        let mut bm = ArenaBump::new(mtp_arena, mtp_cap);
        let pre_fc_norm_emb = upload_bf16(cuda, &mut bm, "mtp.pre_fc_norm_embedding.weight", &store.weight_map, &shards)?;
        let pre_fc_norm_hid = upload_bf16(cuda, &mut bm, "mtp.pre_fc_norm_hidden.weight", &store.weight_map, &shards)?;
        let fc_w = upload_bf16(cuda, &mut bm, "mtp.fc.weight", &store.weight_map, &shards)?;
        let norm_w = upload_bf16(cuda, &mut bm, "mtp.norm.weight", &store.weight_map, &shards)?;

        let pfx = "mtp.layers.0";
        let in_norm = upload_bf16(cuda, &mut bm, &format!("{pfx}.input_layernorm.weight"), &store.weight_map, &shards)?;
        let post_norm = upload_bf16(cuda, &mut bm, &format!("{pfx}.post_attention_layernorm.weight"), &store.weight_map, &shards)?;
        let gate_up = upload_packed_bf16(
            cuda,
            &mut bm,
            &[format!("{pfx}.mlp.gate_proj.weight"), format!("{pfx}.mlp.up_proj.weight")],
            &store.weight_map,
            &shards,
            &mut staging,
        )?;
        let down = upload_bf16(cuda, &mut bm, &format!("{pfx}.mlp.down_proj.weight"), &store.weight_map, &shards)?;
        let qkv = upload_packed_bf16(
            cuda,
            &mut bm,
            &[
                format!("{pfx}.self_attn.q_proj.weight"),
                format!("{pfx}.self_attn.k_proj.weight"),
                format!("{pfx}.self_attn.v_proj.weight"),
            ],
            &store.weight_map,
            &shards,
            &mut staging,
        )?;
        let o_proj = upload_bf16(cuda, &mut bm, &format!("{pfx}.self_attn.o_proj.weight"), &store.weight_map, &shards)?;
        let q_norm = upload_bf16(cuda, &mut bm, &format!("{pfx}.self_attn.q_norm.weight"), &store.weight_map, &shards)?;
        let k_norm = upload_bf16(cuda, &mut bm, &format!("{pfx}.self_attn.k_norm.weight"), &store.weight_map, &shards)?;
        let layer = PackedLayerPtrs {
            is_linear: false,
            in_norm, post_norm, gate_up, down,
            in_all: 0, out_proj: 0, conv_w: 0, dt_bias: 0, a_log_f32: 0, dn_norm: 0,
            qkv, o_proj, q_norm, k_norm,
        };
        Some(MtpWeights {
            arena: mtp_arena,
            bytes: bm.offset,
            pre_fc_norm_emb,
            pre_fc_norm_hid,
            fc_w,
            layer,
            norm_w,
        })
    } else {
        None
    };

    // Staging + mmaps drop here -> Host RAM immediately drops to ~100 MB!
    Ok(LoadedSplitWeights {
        p1_arena,
        p1_bytes: b1.offset,
        p2_arena,
        p2_bytes: b2.offset,
        embed_w,
        lm_head_w,
        final_norm_w,
        p1_layers,
        p2_layers,
        mtp,
    })
}

