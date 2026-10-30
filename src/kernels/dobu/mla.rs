//! DoBu MLA — FlashMLA-style interface.
use candle_core::{Device, Result as CResult, Tensor, D};

pub const KERNEL_MLA: &str = "DoBu-MLA";
pub const KERNEL_GEMM: &str = "DoBu-GEMM";

#[derive(Debug, Clone, Default)]
pub struct DobuMlaMeta {
    pub num_heads: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
    pub softmax_scale: f64,
}

impl DobuMlaMeta {
    pub fn new(num_heads: usize, qk_dim: usize, v_dim: usize) -> Self {
        Self { num_heads, qk_dim, v_dim, softmax_scale: 1.0 / (qk_dim as f64).sqrt() }
    }
}

#[derive(Debug, Clone)]
pub struct DobuMlaParams {
    pub causal: bool,
    pub softmax_scale: f64,
}

impl Default for DobuMlaParams {
    fn default() -> Self { Self { causal: true, softmax_scale: 1.0 } }
}

pub fn dobu_mla_prefill(q: &Tensor, k: &Tensor, v: &Tensor, params: &DobuMlaParams) -> CResult<Tensor> {
    let (_b, _h, s, _d) = q.dims4()?;
    let mut att = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * params.softmax_scale)?;
    if params.causal && s > 1 {
        let kv_len = k.dim(2)?;
        att = att.broadcast_add(&causal_mask(s, kv_len, q.device())?)?;
    }
    candle_nn::ops::softmax_last_dim(&att)?.matmul(v)
}

pub fn dobu_mla_with_kvcache(
    q: &Tensor, k_cache: &Tensor, v_cache: &Tensor, meta: &DobuMlaMeta, causal: bool,
) -> CResult<Tensor> {
    let (_b, _h, sq, _) = q.dims4()?;
    let mut att = (q.matmul(&k_cache.transpose(D::Minus1, D::Minus2)?)? * meta.softmax_scale)?;
    if causal && sq > 1 {
        let kv_len = k_cache.dim(2)?;
        att = att.broadcast_add(&causal_mask(sq, kv_len, q.device())?)?;
    }
    candle_nn::ops::softmax_last_dim(&att)?.matmul(v_cache)
}

fn causal_mask(q_len: usize, kv_len: usize, device: &Device) -> CResult<Tensor> {
    let mut mask = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        let max_key = kv_len - q_len + i;
        for j in (max_key + 1)..kv_len {
            mask[i * kv_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
}
