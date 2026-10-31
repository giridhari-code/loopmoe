//! DoBu GEMM — tiled CUDA path + Candle/cuBLAS fallback.

use candle_core::{IndexOp, Result as CResult, Tensor};
use crate::kernels::cuda::{tiled_gemm_nn, tiled_gemm_nt};

pub const KERNEL_GEMM: &str = "DoBu-GEMM";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DobuGemmLayout {
    /// A @ B
    Nn,
    /// A @ Bᵀ
    Nt,
    /// Aᵀ @ B
    Tn,
}

#[derive(Debug, Clone)]
pub struct DobuGemmParams {
    pub layout: DobuGemmLayout,
    pub alpha: f64,
    pub beta: f64,
}

impl Default for DobuGemmParams {
    fn default() -> Self {
        Self {
            layout: DobuGemmLayout::Nn,
            alpha: 1.0,
            beta: 0.0,
        }
    }
}

fn matmul_layout(a: &Tensor, b: &Tensor, layout: DobuGemmLayout) -> CResult<Tensor> {
    match layout {
        DobuGemmLayout::Nn => tiled_gemm_nn(a, b),
        DobuGemmLayout::Nt => tiled_gemm_nt(a, b),
        DobuGemmLayout::Tn => {
            // Aᵀ @ B — transpose A then NN tiled
            tiled_gemm_nn(&a.t()?.contiguous()?, b)
        }
    }
}

/// Dense GEMM with optional bias (broadcast).
pub fn dobu_gemm_nn(
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    layout: DobuGemmLayout,
) -> CResult<Tensor> {
    let out = matmul_layout(a, b, layout)?;
    match bias {
        Some(c) => out.broadcast_add(c),
        None => Ok(out),
    }
}

/// Dense GEMM with alpha / beta (C = alpha * op(A) @ op(B) + beta * C).
pub fn dobu_gemm(
    a: &Tensor,
    b: &Tensor,
    c: Option<&Tensor>,
    params: &DobuGemmParams,
) -> CResult<Tensor> {
    let mut out = matmul_layout(a, b, params.layout)?;
    if (params.alpha - 1.0).abs() > 1e-6 {
        out = (out * params.alpha)?;
    }
    if let Some(c) = c {
        if params.beta.abs() > 1e-12 {
            out = out.add(&(c * params.beta)?)?;
        }
    }
    Ok(out)
}

/// Grouped / expert GEMM used by MoE.
pub fn dobu_grouped_gemm(
    tokens: &Tensor,
    weights: &Tensor,
    offsets: &[usize],
) -> CResult<Tensor> {
    let (_e, _k, n) = weights.dims3()?;
    let total = tokens.dim(0)?;
    let device = tokens.device();
    let dtype = tokens.dtype();

    if offsets.len() < 2 {
        return Tensor::zeros((total, n), dtype, device);
    }

    let mut outs: Vec<(usize, usize, Tensor)> = Vec::with_capacity(offsets.len() - 1);
    for ei in 0..offsets.len().saturating_sub(1) {
        let start = offsets[ei];
        let end = offsets[ei + 1];
        if end <= start {
            continue;
        }
        let chunk = tokens.narrow(0, start, end - start)?;
        let w = weights.i(ei)?;
        // Prefer tiled GEMM per expert chunk
        outs.push((start, end, tiled_gemm_nn(&chunk, &w)?));
    }

    if outs.is_empty() {
        return Tensor::zeros((total, n), dtype, device);
    }

    let mut parts = Vec::new();
    let mut cursor = 0usize;
    for (start, end, piece) in outs {
        if start > cursor {
            parts.push(Tensor::zeros((start - cursor, n), dtype, device)?);
        }
        parts.push(piece);
        cursor = end;
    }
    if cursor < total {
        parts.push(Tensor::zeros((total - cursor, n), dtype, device)?);
    }
    Tensor::cat(&parts, 0)
}
