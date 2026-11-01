//! Qwen-style architecture — **planned port**, not wired yet.

use crate::config::ModelConfig;

#[derive(Debug)]
pub struct QwenModel {
    pub config: ModelConfig,
}

impl QwenModel {
    pub fn is_supported() -> bool {
        false
    }

    pub fn roadmap() -> &'static str {
        "QwenModel: GQA + SwiGLU MLP; optional Qwen2 MoE later; InferModel"
    }
}
