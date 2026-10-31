//! ModelConfig + ServerConfig

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_path: String,
    pub max_model_len: usize,
    pub dtype: String,
    pub quantization: Option<String>,
    // ... extend as needed
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub enable_prefix_caching: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".into(),
            port: 8000,
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            enable_prefix_caching: true,
        }
    }
}
