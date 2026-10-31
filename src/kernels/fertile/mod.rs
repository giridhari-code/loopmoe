//! # Fertile
//!
//! CUDA → ARM → CPU. Grouping: device-side argsort (`DeviceGroupPlan`).
//! Capacity-aware load balancing with optional token drop.

mod backend;
mod dispatch;
mod group_plan;
mod cuda_backend;
mod arm_backend;

pub use backend::{select_backend, BackendInfo, BackendKind};
pub use dispatch::{FertileDispatch, FertileExpert, FertileMoeParams, fertile_moe};
pub use group_plan::{topk_last, DeviceGroupPlan, LoadStats};

pub const KERNEL_NAME: &str = "Fertile";
