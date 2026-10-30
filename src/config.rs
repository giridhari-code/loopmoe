use serde::{Deserialize, Serialize};

/// Configuration for a Looped MoE model.
/// Inspired by DeepInfer scheduling + modern looped transformer designs (Foil / LOOM style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Core dimensions
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,   // GQA
    pub head_dim: usize,

    // Looped Transformer
    /// Number of distinct physical layers in the shared block
    pub num_layers: usize,
    /// How many times the shared block is applied (loop count)
    /// Default 24 as requested
    pub loop_count: usize,

    // MoE
    pub num_experts: usize,
    pub num_experts_per_tok: usize,   // top-k
    pub moe_intermediate_size: usize,
    pub norm_topk_prob: bool,

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

            num_layers: 8,          // physical layers in the shared block
            loop_count: 24,         // requested default

            num_experts: 64,
            num_experts_per_tok: 6,
            moe_intermediate_size: 1408,
            norm_topk_prob: true,

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

    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&data)?;
        Ok(cfg)
    }
}
