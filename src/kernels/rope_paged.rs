//! Paged RoPE entry — dense RoPE is `model/rope.rs`.

use candle_core::{Result as CResult, Tensor};

pub fn rope_paged(_x: &Tensor, _cos: &Tensor, _sin: &Tensor) -> CResult<Tensor> {
    candle_core::bail!("rope_paged: not implemented — use model::rope")
}
