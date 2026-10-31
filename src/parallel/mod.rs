//! Parallelism for Sentinel — single-GPU + multi-GPU.
//!
//! Inspired by DeepEP (expert parallel all-to-all) and standard TP/DP.
//! Current backend is pure Candle. When NCCL / DeepEP FFI is linked, the same
//! traits can dispatch to real device collectives.

pub mod config;
pub mod mesh;
pub mod expert_parallel;
pub mod collectives;

pub use config::{ParallelConfig, ParallelismKind};
pub use mesh::{DeviceMesh, RankInfo};
pub use expert_parallel::{ExpertParallelPlan, ExpertShard};
pub use collectives::{AllToAllStub, DispatchMeta, CombineMeta};
