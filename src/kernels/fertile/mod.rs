//! # Fertile
//!
//! Backends: CUDA (cudarc via candle) → ARM (aarch64) → CPU.

mod backend;
mod dispatch;
mod cuda_backend;
mod arm_backend;

pub use backend::{select_backend, BackendInfo, BackendKind};
pub use dispatch::{FertileDispatch, FertileExpert, FertileMoeParams, fertile_moe};

pub const KERNEL_NAME: &str = "Fertile";
