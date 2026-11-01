use serde::{Deserialize, Serialize};
use crate::config::ModelConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrainStage {
    Pretrain,
    Midtrain,
    Sft,
    Dpo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    pub stage: TrainStage,
    #[serde(default)]
    pub model: ModelConfig,
    pub data_path: String,
    #[serde(default = "default_out")]
    pub output_dir: String,
    #[serde(default)]
    pub init_from: Option<String>,
    #[serde(default = "default_steps")]
    pub max_steps: usize,
    #[serde(default = "default_bs")]
    pub batch_size: usize,
    #[serde(default = "default_seq")]
    pub max_seq_len: usize,
    #[serde(default = "default_lr")]
    pub lr: f64,
    #[serde(default = "default_wd")]
    pub weight_decay: f64,
    #[serde(default = "default_accum")]
    pub grad_accum: usize,
    #[serde(default = "default_log")]
    pub log_every: usize,
    #[serde(default = "default_save")]
    pub save_every: usize,
    #[serde(default = "default_beta")]
    pub dpo_beta: f64,
    #[serde(default = "default_aux")]
    pub moe_aux_coef: f64,
    /// Linear warmup steps before cosine decay. 0 = no warmup.
    #[serde(default = "default_warmup")]
    pub warmup_steps: usize,
    /// Global L2 grad-norm clip. 0 = disabled.
    #[serde(default = "default_clip")]
    pub max_grad_norm: f64,
    /// Cosine floor as fraction of peak LR (default 0.1).
    #[serde(default = "default_min_lr_ratio")]
    pub min_lr_ratio: f64,
}

fn default_out() -> String { "outputs/train".into() }
fn default_steps() -> usize { 10_000 }
fn default_bs() -> usize { 1 }
fn default_seq() -> usize { 512 }
fn default_lr() -> f64 { 3e-4 }
fn default_wd() -> f64 { 0.1 }
fn default_accum() -> usize { 1 }
fn default_log() -> usize { 10 }
fn default_save() -> usize { 1000 }
fn default_beta() -> f64 { 0.1 }
fn default_aux() -> f64 { 0.001 }
fn default_warmup() -> usize { 100 }
fn default_clip() -> f64 { 1.0 }
fn default_min_lr_ratio() -> f64 { 0.1 }

impl TrainConfig {
    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    pub fn pretrain_defaults() -> Self {
        Self {
            stage: TrainStage::Pretrain,
            model: ModelConfig::sentinel_v1(),
            data_path: "data/pretrain.jsonl".into(),
            output_dir: "outputs/pretrain".into(),
            init_from: None,
            max_steps: 100_000,
            batch_size: 1,
            max_seq_len: 2048,
            lr: 3e-4,
            weight_decay: 0.1,
            grad_accum: 8,
            log_every: 10,
            save_every: 1000,
            dpo_beta: 0.1,
            moe_aux_coef: 0.001,
            warmup_steps: 1000,
            max_grad_norm: 1.0,
            min_lr_ratio: 0.1,
        }
    }
}
