//! Collectives — DeepEP-style dispatch / combine (Orbit O5).
//!
//! - `ep_size == 1`: identity
//! - `ep_size > 1` + top-k expert ids: **logical pack-by-rank** (reorder tokens)
//! - Device NCCL/DeepEP transfer: still stub (same process)

use candle_core::{IndexOp, Result as CandleResult, Tensor, DType};
use super::expert_parallel::ExpertParallelPlan;
use crate::kernels::fertile::LoadStats;

#[derive(Debug, Clone)]
pub struct DispatchMeta {
    pub tokens_per_rank: Vec<usize>,
    pub offsets: Vec<usize>,
    pub expert_ids: Option<Vec<i32>>,
    pub load_stats: Option<LoadStats>,
    /// Maps packed row → original token index (for combine).
    pub inverse_perm: Option<Vec<usize>>,
}

impl DispatchMeta {
    pub fn identity(n_tokens: usize) -> Self {
        Self {
            tokens_per_rank: vec![n_tokens],
            offsets: vec![0, n_tokens],
            expert_ids: None,
            load_stats: None,
            inverse_perm: None,
        }
    }

    pub fn with_load_stats(mut self, stats: LoadStats) -> Self {
        self.load_stats = Some(stats);
        self
    }
}

#[derive(Debug, Clone)]
pub struct CombineMeta {
    pub tokens_per_rank: Vec<usize>,
    pub offsets: Vec<usize>,
    pub inverse_perm: Option<Vec<usize>>,
}

impl CombineMeta {
    pub fn from_dispatch(d: &DispatchMeta) -> Self {
        Self {
            tokens_per_rank: d.tokens_per_rank.clone(),
            offsets: d.offsets.clone(),
            inverse_perm: d.inverse_perm.clone(),
        }
    }
}

/// Map global expert id → EP rank (contiguous shards).
pub fn expert_to_rank(expert_id: usize, n_routed: usize, ep_size: usize) -> usize {
    let ep = ep_size.max(1);
    if n_routed == 0 {
        return 0;
    }
    // Match ExpertParallelPlan::from_mesh sharding
    let base = n_routed / ep;
    let rem = n_routed % ep;
    let mut start = 0;
    for r in 0..ep {
        let count = base + if r < rem { 1 } else { 0 };
        let end = start + count;
        if expert_id >= start && expert_id < end {
            return r;
        }
        start = end;
    }
    ep - 1
}

/// Pack token rows by destination EP rank using primary top-1 expert id.
/// `tokens`: [T, H], `topk_ids`: [T, K] i32/u32 expert ids.
pub fn pack_by_ep_rank(
    tokens: &Tensor,
    topk_ids: &Tensor,
    n_routed: usize,
    ep_size: usize,
) -> CandleResult<(Tensor, DispatchMeta)> {
    let t = tokens.dim(0)?;
    let ep = ep_size.max(1);
    if ep <= 1 || t == 0 {
        return Ok((tokens.clone(), DispatchMeta::identity(t)));
    }

    // primary expert = first column
    let ids = if topk_ids.dims().len() == 1 {
        topk_ids.to_vec1::<i64>().or_else(|_| {
            topk_ids
                .to_dtype(DType::I64)?
                .to_vec1::<i64>()
        })?
    } else {
        let col0 = topk_ids.i((.., 0))?;
        col0.to_dtype(DType::I64)?.to_vec1::<i64>()?
    };

    let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); ep];
    for (row, &eid) in ids.iter().enumerate() {
        let rank = expert_to_rank(eid.max(0) as usize, n_routed, ep);
        buckets[rank].push(row);
    }

    let mut tokens_per_rank = Vec::with_capacity(ep);
    let mut offsets = vec![0usize];
    let mut perm = Vec::with_capacity(t);
    for b in &buckets {
        tokens_per_rank.push(b.len());
        perm.extend_from_slice(b);
        offsets.push(perm.len());
    }

    // inverse: packed_idx → original
    let mut inverse = vec![0usize; t];
    for (packed_i, &orig) in perm.iter().enumerate() {
        if orig < t {
            inverse[packed_i] = orig;
        }
    }

    // Gather rows — CPU index path (portable)
    let h = tokens.dim(1)?;
    let flat = tokens.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut packed = vec![0f32; t * h];
    for (new_i, &old_i) in perm.iter().enumerate() {
        let src = old_i * h;
        let dst = new_i * h;
        packed[dst..dst + h].copy_from_slice(&flat[src..src + h]);
    }
    let out = Tensor::from_vec(packed, (t, h), tokens.device())?
        .to_dtype(tokens.dtype())?;

    let meta = DispatchMeta {
        tokens_per_rank,
        offsets,
        expert_ids: Some(ids.iter().map(|&x| x as i32).collect()),
        load_stats: None,
        inverse_perm: Some(inverse),
    };
    Ok((out, meta))
}

/// Scatter packed outputs back to original token order.
pub fn unpack_combine(expert_out: &Tensor, meta: &CombineMeta) -> CandleResult<Tensor> {
    let Some(ref inv) = meta.inverse_perm else {
        return Ok(expert_out.clone());
    };
    let t = expert_out.dim(0)?;
    let h = expert_out.dim(1)?;
    if inv.len() != t {
        return Ok(expert_out.clone());
    }
    let flat = expert_out
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut out = vec![0f32; t * h];
    // inv[packed] = original → we need original order: for each packed row, write to inv[p]
    // Actually inverse_perm[packed_i] = original_i, so out[original] = packed[packed_i]
    for (packed_i, &orig) in inv.iter().enumerate() {
        if orig >= t {
            continue;
        }
        let src = packed_i * h;
        let dst = orig * h;
        out[dst..dst + h].copy_from_slice(&flat[src..src + h]);
    }
    Tensor::from_vec(out, (t, h), expert_out.device())?.to_dtype(expert_out.dtype())
}

pub struct AllToAllStub {
    pub ep_size: usize,
    pub ep_rank: usize,
    pub n_routed: usize,
}

impl AllToAllStub {
    pub fn from_plan(plan: &ExpertParallelPlan) -> Self {
        Self {
            ep_size: plan.shard.ep_size,
            ep_rank: plan.shard.ep_rank,
            n_routed: plan.n_routed_experts,
        }
    }

    pub fn is_identity(&self) -> bool {
        self.ep_size <= 1
    }

    /// Dispatch: logical pack-by-rank when topk provided and ep_size > 1.
    pub fn dispatch(
        &self,
        tokens: &Tensor,
        topk_ids: Option<&Tensor>,
        _topk_w: Option<&Tensor>,
    ) -> CandleResult<(Tensor, DispatchMeta)> {
        let n = tokens.dim(0)?;
        if self.is_identity() {
            return Ok((tokens.clone(), DispatchMeta::identity(n)));
        }
        if let Some(ids) = topk_ids {
            tracing::debug!(
                ep_size = self.ep_size,
                ep_rank = self.ep_rank,
                "EP dispatch: pack_by_ep_rank (logical, no NCCL)"
            );
            return pack_by_ep_rank(tokens, ids, self.n_routed, self.ep_size);
        }
        tracing::debug!(
            ep_size = self.ep_size,
            "EP dispatch identity (no topk ids)"
        );
        Ok((tokens.clone(), DispatchMeta::identity(n)))
    }

    pub fn combine(&self, expert_out: &Tensor, meta: &CombineMeta) -> CandleResult<Tensor> {
        if self.is_identity() || meta.inverse_perm.is_none() {
            return Ok(expert_out.clone());
        }
        unpack_combine(expert_out, meta)
    }
}
