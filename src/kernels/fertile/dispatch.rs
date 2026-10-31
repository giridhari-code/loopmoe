//! Fertile dispatch — CUDA (sequential) / ARM+CPU (parallel experts).

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
    /// Capacity factor for load balancing (DeepSeek / Switch style).
    /// <= 0.0 disables capacity limit (keeps all tokens).
    /// Typical values: 1.0 – 1.5. Higher = fewer drops, more imbalance risk.
    pub capacity_factor: f32,
}

impl Default for FertileMoeParams {
    fn default() -> Self {
        Self {
            num_experts_per_tok: 6,
            routed_scaling_factor: 1.0,
            norm_topk_prob: true,
            use_sigmoid: false,
            capacity_factor: 1.25,
        }
    }
}

pub trait FertileExpert: Sync {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor>;
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
            BackendKind::Arm | BackendKind::Cpu => {
                fertile_moe_arm(x_flat, gate, routed, shared, params)
            }
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
