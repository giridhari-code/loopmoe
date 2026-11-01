//! Sentinel v1.0 Rust training: pretrain → midtrain → sft → dpo

pub mod config;
pub mod data;
pub mod model;
pub mod checkpoint;
pub mod optim;
pub mod pretrain;
pub mod midtrain;
pub mod sft;
pub mod dpo;

pub use config::TrainConfig;
pub use model::SentinelTrainModel;
pub use optim::{LrSchedule, backward_step_clipped, clip_grad_norm};
