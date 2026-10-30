//! **DoBu** — Sentinel fused compute kernels.
//! DeepGEMM → dobu::gemm | FlashMLA → dobu::mla

pub mod gemm;
pub mod mla;

pub use gemm::{dobu_gemm_nn, dobu_grouped_gemm, DobuGemmLayout};
pub use mla::{dobu_mla_prefill, dobu_mla_with_kvcache, DobuMlaMeta, DobuMlaParams, KERNEL_MLA, KERNEL_GEMM};

pub const KERNEL_SUITE: &str = "DoBu";
