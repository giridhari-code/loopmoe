use serde::{Deserialize, Serialize};

/// **Sentinel v1.0** model configuration.
/// MoE fields follow DeepInfer / DeepSeek HF conventions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_name: String,
    pub model_type: String,

    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,

    pub num_layers: usize,
    pub loop_count: usize,

    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub routed_scaling_factor: f32,
    pub scoring_func: String,
    pub norm_topk_prob: bool,
    pub topk_method: String,

    pub max_position_embeddings: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,

    pub rms_norm_eps: f32,
    pub hidden_act: String,

    pub tie_word_embeddings: bool,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,

    #[serde(default)]
    pub use_mla: bool,
    #[serde(default = "d_q_lora")]
    pub q_lora_rank: usize,
    #[serde(default = "d_kv_lora")]
    pub kv_lora_rank: usize,
    #[serde(default = "d_nope")]
    pub qk_nope_head_dim: usize,
    #[serde(default = "d_rope")]
    pub qk_rope_head_dim: usize,
    #[serde(default = "d_vdim")]
    pub v_head_dim: usize,
}

fn d_q_lora() -> usize { 1536 }
fn d_kv_lora() -> usize { 512 }
fn d_nope() -> usize { 128 }
fn d_rope() -> usize { 64 }
fn d_vdim() -> usize { 128 }

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            model_name: "Sentinel-v1.0".to_string(),
            model_type: "sentinel_looped_moe".to_string(),

            vocab_size: 128_256,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,

            num_layers: 8,
            loop_count: 24,

            n_routed_experts: 64,
            n_shared_experts: 2,
            num_experts_per_tok: 6,
            moe_intermediate_size: 1408,
            routed_scaling_factor: 1.0,
            scoring_func: "softmax".to_string(),
            norm_topk_prob: true,
            topk_method: "greedy".to_string(),

            max_position_embeddings: 131_072,
            max_seq_len: 4096,
            rope_theta: 500_000.0,

            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),

            tie_word_embeddings: false,
            use_sliding_window: false,
            sliding_window: None,

            use_mla: false,
            q_lora_rank: d_q_lora(),
            kv_lora_rank: d_kv_lora(),
            qk_nope_head_dim: d_nope(),
            qk_rope_head_dim: d_rope(),
            v_head_dim: d_vdim(),
        }
    }
}

impl ModelConfig {
    pub const MODEL_ID: &'static str = "Sentinel-v1.0";

    pub fn effective_depth(&self) -> usize {
        self.num_layers * self.loop_count
    }

    pub fn total_experts(&self) -> usize {
        self.n_routed_experts + self.n_shared_experts
    }

    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let data = std::fs::read_to_string(path)?;
        let mut cfg: Self = serde_json::from_str(&data)?;
        if cfg.model_name.is_empty() {
            cfg.model_name = Self::MODEL_ID.to_string();
        }
        Ok(cfg)
    }

    pub fn sentinel_v1() -> Self {
        Self::default()
    }
}
