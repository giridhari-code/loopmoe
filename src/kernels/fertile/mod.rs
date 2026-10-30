//! # Fertile
//!
//! Model-specific compute kernel for LoopMoE
//! (Looped Transformer + DeepSeek-style Shared/Routed MoE).
//!
//! Design goals:
//! - One clear entry for the MoE stage
//! - Expert-grouped dispatch: each routed expert runs at most once
//! - Shared experts always-on, outside the routed path
//! - Same contract for a future CUDA backend

mod dispatch;

pub use dispatch::{FertileDispatch, FertileExpert, FertileMoeParams, fertile_moe};

/// Kernel identity for logs / profiling.
pub const KERNEL_NAME: &str = "Fertile";
