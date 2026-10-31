//! DoBu GEMM — tiled CUDA + FP8 weight path + Candle fallback.

use candle_core::{IndexOp, Result as CResult, Tensor};
use crate::kernels::cuda::{gemm_nn_fp8_weight, quantize_to_fp8_e4m3, tiled_gemm_nn, tiled_gemm_nt};

pub const KERNEL_GEMM: &str = "DoBu-GEMM";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DobuGemmLayout {
    Nn,
    Nt,
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
        DobuGemmLayout::Tn => tiled_gemm_nn(&a.t()?.contiguous()?, b),
    }
}

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

/// FP8 weight-only GEMM: `activations @ dequant(fp8_weights) * scale`.
/// `b_fp8` is U8 E4M3 storage (K, N).
pub fn dobu_gemm_fp8(
    activations: &Tensor,
    b_fp8: &Tensor,
    scale: f32,
    bias: Option<&Tensor>,
) -> CResult<Tensor> {
    let out = gemm_nn_fp8_weight(activations, b_fp8, scale)?;
    match bias {
        Some(b) => out.broadcast_add(b),
        None => Ok(out),
    }
}

/// Quantize a weight matrix to FP8 E4M3 + scale (for offline pack / load).
pub fn dobu_quantize_weight_fp8(weight: &Tensor) -> CResult<(Tensor, f32)> {
    quantize_to_fp8_e4m3(weight)
}

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
