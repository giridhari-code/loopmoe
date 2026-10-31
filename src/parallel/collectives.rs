//! Collectives stubs — DeepEP-style dispatch / combine.
//!
//! Real implementation will call DeepEP / NCCL. For now we provide:
//! - Single-rank identity (no communication)
//! - Metadata structures matching typical EP all-to-all layouts

use candle_core::{Result as CandleResult, Tensor};
use super::expert_parallel::ExpertParallelPlan;

/// Metadata produced by dispatch (token packing per expert / rank).
#[derive(Debug, Clone)]
pub struct DispatchMeta {
    /// Number of tokens sent to each EP rank.
    pub tokens_per_rank: Vec<usize>,
    /// Exclusive prefix sum for packing.
    pub offsets: Vec<usize>,
    /// Optional expert ids for the packed stream.
    pub expert_ids: Option<Vec<i32>>,
}

impl DispatchMeta {
    pub fn identity(n_tokens: usize) -> Self {
        Self {
            tokens_per_rank: vec![n_tokens],
            offsets: vec![0, n_tokens],
            expert_ids: None,
        }
    }
}

/// Metadata for combine (scatter-add back to original token order).
#[derive(Debug, Clone)]
pub struct CombineMeta {
    pub tokens_per_rank: Vec<usize>,
    pub offsets: Vec<usize>,
}

impl CombineMeta {
    pub fn from_dispatch(d: &DispatchMeta) -> Self {
        Self {
            tokens_per_rank: d.tokens_per_rank.clone(),
            offsets: d.offsets.clone(),
        }
    }
}

/// All-to-all stub. Single rank = identity.
pub struct AllToAllStub {
    pub ep_size: usize,
    pub ep_rank: usize,
}

impl AllToAllStub {
    pub fn from_plan(plan: &ExpertParallelPlan) -> Self {
        Self {
            ep_size: plan.shard.ep_size,
            ep_rank: plan.shard.ep_rank,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.ep_size <= 1
    }

    /// Dispatch tokens to expert ranks.
    /// Single-GPU: returns input unchanged + identity meta.
    pub fn dispatch(
        &self,
        tokens: &Tensor,
        _topk_ids: Option<&Tensor>,
        _topk_w: Option<&Tensor>,
    ) -> CandleResult<(Tensor, DispatchMeta)> {
        let n = tokens.dim(0)?;
        if self.is_identity() {
            return Ok((tokens.clone(), DispatchMeta::identity(n)));
        }
        // Multi-GPU path: placeholder — real DeepEP would pack by destination rank.
        // For now still identity so single-process multi-config does not break.
        tracing::debug!(
            ep_size = self.ep_size,
            ep_rank = self.ep_rank,
            "AllToAllStub::dispatch identity (DeepEP FFI not linked)"
        );
        Ok((tokens.clone(), DispatchMeta::identity(n)))
    }

    /// Combine expert outputs back to token order.
    pub fn combine(
        &self,
        expert_out: &Tensor,
        meta: &CombineMeta,
    ) -> CandleResult<Tensor> {
        if self.is_identity() {
            return Ok(expert_out.clone());
        }
        tracing::debug!(
            ep_size = self.ep_size,
            ep_rank = self.ep_rank,
            n = meta.tokens_per_rank.get(self.ep_rank).copied().unwrap_or(0),
            "AllToAllStub::combine identity (DeepEP FFI not linked)"
        );
        Ok(expert_out.clone())
    }
}
