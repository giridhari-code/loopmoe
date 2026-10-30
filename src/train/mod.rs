//! Sentinel v1.0 Rust training: pretrain → midtrain → sft → dpo

pub mod config;
pub mod data;
pub mod model;
pub mod checkpoint;
pub mod pretrain;
pub mod midtrain;
pub mod sft;
pub mod dpo;

pub use config::TrainConfig;
pub use model::SentinelTrainModel;
