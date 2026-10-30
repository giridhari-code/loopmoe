// Placeholder for temperature / top-p / top-k sampling
// Will be expanded later (DeepInfer has deterministic sampling + structured output)

use candle_core::{Result as CandleResult, Tensor};

pub fn greedy(logits: &Tensor) -> CandleResult<u32> {
    logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()
}
