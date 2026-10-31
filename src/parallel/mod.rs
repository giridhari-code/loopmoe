//! Parallelism for Sentinel — single-GPU + multi-GPU EP/TP/DP.
//!
//! O5: logical EP pack/unpack by expert→rank; NCCL/DeepEP device path remains hookable.

pub mod config;
pub mod mesh;
pub mod expert_parallel;
pub mod collectives;

pub use config::{ParallelConfig, ParallelismKind};
pub use mesh::{DeviceMesh, RankInfo};
pub use expert_parallel::{ExpertParallelPlan, ExpertShard};
pub use collectives::{AllToAllStub, CombineMeta, DispatchMeta, pack_by_ep_rank, unpack_combine, expert_to_rank};
