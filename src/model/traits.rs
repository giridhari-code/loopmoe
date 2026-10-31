//! Shared inference surface for Orbit / engine (model-agnostic).

use candle_core::{Device, Result as CandleResult, Tensor};

use crate::cache::RuntimeCaches;
use crate::config::ModelConfig;

/// Minimal contract every generative backbone implements for this runtime.
pub trait InferModel {
    fn config(&self) -> &ModelConfig;
    fn device(&self) -> &Device;

    /// Causal forward. `input_ids`: [B, S]. Returns logits for last position or [B,V].
    fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        caches: Option<&mut RuntimeCaches>,
    ) -> CandleResult<Tensor>;

    /// O2.1 batched decode: one token per sequence, logits [B, vocab].
    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        caches: &mut [&mut RuntimeCaches],
    ) -> CandleResult<Tensor>;
}
