//! **DoBu** — Sentinel fused compute kernels.
//!
//! Naming / inspiration:
//! - DeepGEMM  → `dobu::gemm`
//! - FlashMLA  → `dobu::mla`
//!
//! Current implementation is pure Candle (portable). Interfaces are shaped so
//! that a future CUDA/FFI backend (DeepGEMM / FlashMLA) can drop in behind the
//! same function signatures.

pub mod gemm;
pub mod mla;

pub use gemm::{dobu_gemm_nn, dobu_grouped_gemm, DobuGemmLayout, DobuGemmParams};
pub use mla::{
    dobu_mla_prefill, dobu_mla_with_kvcache, DobuMlaMeta, DobuMlaParams, KERNEL_MLA,
};

pub const KERNEL_SUITE: &str = "DoBu";
pub const KERNEL_GEMM: &str = "DoBu-GEMM";
