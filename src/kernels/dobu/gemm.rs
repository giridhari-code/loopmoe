//! DoBu GEMM — DeepGEMM-style interface.
use candle_core::{IndexOp, Result as CResult, Tensor};

pub const KERNEL_GEMM: &str = "DoBu-GEMM";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DobuGemmLayout { Nn, Nt }

pub fn dobu_gemm_nn(a: &Tensor, b: &Tensor, bias: Option<&Tensor>, layout: DobuGemmLayout) -> CResult<Tensor> {
    let out = match layout {
        DobuGemmLayout::Nn => a.matmul(b)?,
        DobuGemmLayout::Nt => a.matmul(&b.t()?)?,
    };
    match bias { Some(c) => out.broadcast_add(c), None => Ok(out) }
}

pub fn dobu_grouped_gemm(tokens: &Tensor, weights: &Tensor, offsets: &[usize]) -> CResult<Tensor> {
    let (_e, _k, n) = weights.dims3()?;
    let total = tokens.dim(0)?;
    let device = tokens.device();
    let dtype = tokens.dtype();
    let mut outs = Vec::new();
    for ei in 0..offsets.len().saturating_sub(1) {
        let start = offsets[ei];
        let end = offsets[ei + 1];
        if end <= start { continue; }
        let chunk = tokens.narrow(0, start, end - start)?;
        let w = weights.i(ei)?;
        outs.push((start, end, chunk.matmul(&w)?));
    }
    if outs.is_empty() { return Tensor::zeros((total, n), dtype, device); }
    let mut parts = Vec::new();
    let mut cursor = 0usize;
    for (start, end, piece) in outs {
        if start > cursor { parts.push(Tensor::zeros((start - cursor, n), dtype, device)?); }
        parts.push(piece);
        cursor = end;
    }
    if cursor < total { parts.push(Tensor::zeros((total - cursor, n), dtype, device)?); }
    Tensor::cat(&parts, 0)
}
