//! # Fertile
//!
//! CUDA → ARM → CPU. Grouping: device-side argsort (`DeviceGroupPlan`).

mod backend;
mod dispatch;
mod group_plan;
mod cuda_backend;
mod arm_backend;

pub use backend::{select_backend, BackendInfo, BackendKind};
pub use dispatch::{FertileDispatch, FertileExpert, FertileMoeParams, fertile_moe};
pub use group_plan::DeviceGroupPlan;

pub const KERNEL_NAME: &str = "Fertile";
