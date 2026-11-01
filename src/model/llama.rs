//! Llama-style dense transformer — **planned port**, not wired to Orbit yet.
//!
//! Sentinel production model is [`super::LoopedMoE`]. This module exists so
//! multi-architecture support can land without inventing a second engine.

use crate::config::ModelConfig;

/// Placeholder for future Llama-2/3 dense load path.
#[derive(Debug)]
pub struct LlamaModel {
    pub config: ModelConfig,
}

impl LlamaModel {
    pub fn is_supported() -> bool {
        false
    }

    pub fn roadmap() -> &'static str {
        "LlamaModel: share Attention/RmsNorm/RoPE with LoopedMoE; dense MLP instead of MoE; InferModel impl"
    }
}
