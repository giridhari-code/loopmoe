//! Paged RMSNorm entry — use `cuda::rmsnorm` / sentinel.cu for dense path.

use candle_core::{Result as CResult, Tensor};

pub fn rmsnorm_paged(_x: &Tensor, _weight: &Tensor, _eps: f64) -> CResult<Tensor> {
    candle_core::bail!("rmsnorm_paged: not implemented — use cuda::rmsnorm")
}
