//! **Tesla** — Sentinel speculative decoding (DSpark / DeepInfer MTP inspired).
//!
//! - Chain or **tree draft** (branched, survival-ranked)
//! - **Chunk verify** (one sequence forward / heavy matmuls once)
//! - **MTP varlen** confidence + accept-rate scheduled verification
//! - **DoBu GEMM fuse** helpers for linear / QK / AV paths

pub mod config;
pub mod confidence;
pub mod draft;
pub mod fuse;
pub mod mtp;
pub mod tree;
pub mod verify;

pub use config::TeslaConfig;
pub use confidence::{
    position_confidence, position_confidence_margin, schedule_verify_len,
    schedule_verify_len_smart,
};
pub use draft::draft_tokens;
pub use fuse::{fuse_av, fuse_gemm, fuse_linear_nn, fuse_linear_nt, fuse_qk_scores};
pub use mtp::MtpState;
pub use tree::{draft_tree, TreePath};
pub use verify::{verify_and_accept, TeslaStepStats};
