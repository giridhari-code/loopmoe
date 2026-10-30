//! Fertile dispatch — routes to CUDA / ARM / CPU backends.

use candle_core::{Result as CandleResult, Tensor};
use candle_nn::Module;

use super::backend::{select_backend, BackendKind};
use super::arm_backend::fertile_moe_arm;
use super::cuda_backend::fertile_moe_cuda;

#[derive(Clone, Copy, Debug)]
pub struct FertileMoeParams {
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub use_sigmoid: bool,
}

pub trait FertileExpert {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor>;
}

pub(crate) struct ExpertBuckets {
    pub token_indices: Vec<Vec<u32>>,
    pub weights: Vec<Vec<f32>>,
}

impl ExpertBuckets {
    pub(crate) fn from_topk(
        topk_ids: &[i64],
        topk_w: &[f32],
        num_tokens: usize,
        k: usize,
        n_routed: usize,
        scale: f32,
    ) -> Self {
        let mut token_indices = vec![Vec::new(); n_routed];
        let mut weights = vec![Vec::new(); n_routed];
        for tok in 0..num_tokens {
            for slot in 0..k {
                let idx = tok * k + slot;
                let e = topk_ids[idx] as usize;
                if e < n_routed {
                    token_indices[e].push(tok as u32);
                    weights[e].push(topk_w[idx] * scale);
                }
            }
        }
        Self { token_indices, weights }
    }
}

pub struct FertileDispatch;

impl FertileDispatch {
    pub fn forward<E: FertileExpert>(
        x_flat: &Tensor,
        gate: &impl Module,
        routed: &[E],
        shared: &[E],
        params: FertileMoeParams,
    ) -> CandleResult<Tensor> {
        match select_backend(x_flat.device()) {
            BackendKind::Cuda => fertile_moe_cuda(x_flat, gate, routed, shared, params),
            BackendKind::Arm => fertile_moe_arm(x_flat, gate, routed, shared, params),
            BackendKind::Cpu => fertile_moe_arm(x_flat, gate, routed, shared, params),
        }
    }
}

pub fn fertile_moe<E: FertileExpert>(
    x_flat: &Tensor,
    gate: &impl Module,
    routed: &[E],
    shared: &[E],
    params: FertileMoeParams,
) -> CandleResult<Tensor> {
    FertileDispatch::forward(x_flat, gate, routed, shared, params)
}

pub(crate) fn scatter_add(base: &Tensor, indices: &Tensor, values: &Tensor) -> CandleResult<Tensor> {
    let h = values.dim(1)?;
    let n = indices.dims1()?;
    let indices_exp = indices.unsqueeze(1)?.expand((n, h))?.contiguous()?;
    base.scatter_add(&indices_exp, values, 0)
}
