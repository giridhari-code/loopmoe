//! DoBu MLA — FlashMLA-style interface (Candle backend).
//!
//! Provides prefill and KV-cache decode paths shaped like FlashMLA so a future
//! CUDA backend can replace the bodies without changing call sites.

use candle_core::{Device, Result as CResult, Tensor, D};

pub const KERNEL_MLA: &str = "DoBu-MLA";

#[derive(Debug, Clone, Default)]
pub struct DobuMlaMeta {
    pub num_heads: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
    pub softmax_scale: f64,
}

impl DobuMlaMeta {
    pub fn new(num_heads: usize, qk_dim: usize, v_dim: usize) -> Self {
        Self {
            num_heads,
            qk_dim,
            v_dim,
            softmax_scale: 1.0 / (qk_dim as f64).sqrt(),
        }
    }

    pub fn with_scale(mut self, scale: f64) -> Self {
        self.softmax_scale = scale;
        self
    }
}

#[derive(Debug, Clone)]
pub struct DobuMlaParams {
    pub causal: bool,
    pub softmax_scale: f64,
}

impl Default for DobuMlaParams {
    fn default() -> Self {
        Self {
            causal: true,
            softmax_scale: 1.0,
        }
    }
}

/// Prefill attention: Q, K, V all present (no cache).
/// Shapes expected: (B, H, S, D)
pub fn dobu_mla_prefill(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    params: &DobuMlaParams,
) -> CResult<Tensor> {
    let (_b, _h, s, _d) = q.dims4()?;
    let scale = if params.softmax_scale == 1.0 {
        // caller may leave default; use 1/sqrt(d) from last dim if possible
        let d = q.dim(D::Minus1)? as f64;
        1.0 / d.sqrt()
    } else {
        params.softmax_scale
    };

    let mut att = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;
    if params.causal && s > 1 {
        let kv_len = k.dim(2)?;
        att = att.broadcast_add(&causal_mask(s, kv_len, q.device())?)?;
    }
    candle_nn::ops::softmax_last_dim(&att)?.matmul(v)
}

/// Decode / incremental attention against a KV cache.
/// Q: (B, H, Sq, D)   K/V cache: (B, H, Skv, D)
pub fn dobu_mla_with_kvcache(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    meta: &DobuMlaMeta,
    causal: bool,
) -> CResult<Tensor> {
    let (_b, _h, sq, _) = q.dims4()?;
    let scale = meta.softmax_scale;

    let mut att = (q.matmul(&k_cache.transpose(D::Minus1, D::Minus2)?)? * scale)?;
    if causal && sq > 1 {
        let kv_len = k_cache.dim(2)?;
        att = att.broadcast_add(&causal_mask(sq, kv_len, q.device())?)?;
    }
    candle_nn::ops::softmax_last_dim(&att)?.matmul(v_cache)
}

/// Causal mask: -inf above the diagonal (aligned to the end of the KV sequence).
fn causal_mask(q_len: usize, kv_len: usize, device: &Device) -> CResult<Tensor> {
    let mut mask = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        // Keys that are "in the future" relative to this query position.
        let max_key = kv_len.saturating_sub(q_len).saturating_add(i);
        for j in (max_key + 1)..kv_len {
            mask[i * kv_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
}
