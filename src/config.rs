use serde::{Deserialize, Serialize};

/// Configuration for a Looped MoE model.
/// MoE fields follow DeepInfer / DeepSeek conventions
/// (see deepinfer/hugging_face/config.py → _build_moe).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Core dimensions
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize, // GQA
    pub head_dim: usize,

    // Looped Transformer
    /// Number of distinct physical layers in the shared block
    pub num_layers: usize,
    /// How many times the shared block is applied (loop count)
    pub loop_count: usize,

    // MoE (DeepSeek / DeepInfer style)
    /// Routed experts (selected by top-k router)
    pub n_routed_experts: usize,
    /// Shared experts (always run on every token)
    pub n_shared_experts: usize,
    /// Top-k over routed experts
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    /// Multiply routed expert output by this factor
    pub routed_scaling_factor: f32,
    /// "softmax" or "sigmoid"
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    /// e.g. "greedy" / "group_limited_greedy" (Phase 1 uses greedy top-k)
    pub topk_method: String,

    // RoPE / positional
    pub max_position_embeddings: usize,
    pub rope_theta: f32,

    // Norm
    pub rms_norm_eps: f32,

    // Misc
    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,

            num_layers: 8,
            loop_count: 24,

            // DeepSeek-style defaults
            n_routed_experts: 64,
            n_shared_experts: 2,
            num_experts_per_tok: 6,
            moe_intermediate_size: 1408,
            routed_scaling_factor: 1.0,
            scoring_func: "softmax".to_string(),
            norm_topk_prob: true,
            topk_method: "greedy".to_string(),

            max_position_embeddings: 131_072,
            rope_theta: 500_000.0,

            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window: None,
        }
    }
}

impl ModelConfig {
    /// Effective depth = num_layers * loop_count
    pub fn effective_depth(&self) -> usize {
        self.num_layers * self.loop_count
    }

    /// Total experts stored in the checkpoint (routed + shared)
    pub fn total_experts(&self) -> usize {
        self.n_routed_experts + self.n_shared_experts
    }

    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&data)?;
        Ok(cfg)
    }
}
