use anyhow::Result;
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct RootConfig {
    pub text_config: TextConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParams {
    pub rope_theta: f64,
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
}

fn default_partial_rotary() -> f64 {
    0.25
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub eos_token_id: i64,
    pub layer_types: Vec<String>,
    pub rope_parameters: RopeParams,
}

impl TextConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let raw = fs::read_to_string(model_dir.join("config.json"))?;
        let root: RootConfig = serde_json::from_str(&raw)?;
        Ok(root.text_config)
    }
}

