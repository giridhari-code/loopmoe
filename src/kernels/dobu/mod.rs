//! **DoBu** — Sentinel fused compute kernels (incl. FP8).

pub mod gemm;
pub mod mla;

pub use gemm::{
    dobu_gemm, dobu_gemm_fp8, dobu_gemm_nn, dobu_grouped_gemm, dobu_quantize_weight_fp8,
    DobuGemmLayout, DobuGemmParams,
};
pub use mla::{
    dobu_mla_prefill, dobu_mla_with_kvcache, DobuMlaMeta, DobuMlaParams, KERNEL_MLA,
};

pub const KERNEL_SUITE: &str = "DoBu";
pub const KERNEL_GEMM: &str = "DoBu-GEMM";
