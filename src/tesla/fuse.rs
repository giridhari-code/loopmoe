//! DoBu GEMM fuse helpers used by Tesla.
//!
//! Keeps Tesla matrix multiplies on the DoBu interface so a future DeepGEMM
//! backend can accelerate draft/verify without changing call sites.

use candle_core::{Result as CResult, Tensor};
use crate::kernels::dobu::{
    dobu_gemm, dobu_gemm_nn, DobuGemmLayout, DobuGemmParams,
};

/// Fused linear: y = x @ Wᵀ + bias  (common lm_head / attention projection shape).
pub fn fuse_linear_nt(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> CResult<Tensor> {
    dobu_gemm_nn(x, weight, bias, DobuGemmLayout::Nt)
}

/// Fused linear: y = x @ W + bias.
pub fn fuse_linear_nn(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> CResult<Tensor> {
    dobu_gemm_nn(x, weight, bias, DobuGemmLayout::Nn)
}

/// Alpha/beta GEMM through DoBu (C = α·A@B + β·C).
pub fn fuse_gemm(
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    alpha: f64,
    beta: f64,
    layout: DobuGemmLayout,
) -> CResult<Tensor> {
    dobu_gemm(
        a,
        b,
        c,
        &DobuGemmParams {
            layout,
            alpha,
            beta,
        },
    )
}

/// Attention scores: Q @ Kᵀ * scale  via DoBu Nt path.
pub fn fuse_qk_scores(q: &Tensor, k: &Tensor, scale: f64) -> CResult<Tensor> {
    let scores = dobu_gemm_nn(q, k, None, DobuGemmLayout::Nt)?;
    if (scale - 1.0).abs() > 1e-12 {
        Ok((scores * scale)?)
    } else {
        Ok(scores)
    }
}

/// Output projection: att @ V via DoBu Nn.
pub fn fuse_av(att: &Tensor, v: &Tensor) -> CResult<Tensor> {
    dobu_gemm_nn(att, v, None, DobuGemmLayout::Nn)
}

/// Batched top-k indices from last dim (for tree branching) — values + ids.
pub fn topk_last_ids(logits: &Tensor, k: usize) -> CResult<(Tensor, Tensor)> {
    crate::kernels::fertile::topk_last(logits, k)
}
