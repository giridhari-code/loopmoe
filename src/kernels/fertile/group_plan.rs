//! Device-side token→expert grouping for Fertile.
//!
//! Supports capacity_factor based load balancing (DeepSeek / Switch style).
//! Excess tokens beyond per-expert capacity are dropped (highest-score kept).

use candle_core::{DType, Result as CandleResult, Tensor, D};
use std::cmp::Ordering;

/// Per-forward load statistics for monitoring imbalance + drops.
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    /// Tokens assigned to each expert after capacity truncation.
    pub tokens_per_expert: Vec<usize>,
    /// Total token-expert assignments before capacity (t * k).
    pub total_assignments: usize,
    /// Assignments that survived capacity.
    pub kept_assignments: usize,
    /// Assignments dropped because expert was over capacity.
    pub dropped_assignments: usize,
    /// capacity used (same for all experts). 0 means unlimited.
    pub capacity: usize,
    /// max tokens on any expert after truncation.
    pub max_load: usize,
    /// min tokens on any expert (can be 0).
    pub min_load: usize,
}

impl LoadStats {
    pub fn imbalance_ratio(&self) -> f32 {
        if self.min_load == 0 {
            if self.max_load == 0 {
                1.0
            } else {
                f32::INFINITY
            }
        } else {
            self.max_load as f32 / self.min_load as f32
        }
    }

    pub fn drop_rate(&self) -> f32 {
        if self.total_assignments == 0 {
            0.0
        } else {
            self.dropped_assignments as f32 / self.total_assignments as f32
        }
    }

    pub fn coefficient_of_variation(&self) -> f32 {
        let n = self.tokens_per_expert.len();
        if n == 0 {
            return 0.0;
        }
        let mean = self.kept_assignments as f32 / n as f32;
        if mean <= 0.0 {
            return 0.0;
        }
        let var: f32 = self
            .tokens_per_expert
            .iter()
            .map(|&c| {
                let d = c as f32 - mean;
                d * d
            })
            .sum::<f32>()
            / n as f32;
        var.sqrt() / mean
    }

    /// True when drop_rate > threshold or imbalance is severe — useful for adaptive capacity.
    pub fn is_unhealthy(&self, drop_threshold: f32, imbalance_threshold: f32) -> bool {
        self.drop_rate() > drop_threshold || self.imbalance_ratio() > imbalance_threshold
    }
}

pub struct DeviceGroupPlan {
    pub sorted_tokens: Tensor,
    pub sorted_weights: Tensor,
    pub expert_offsets: Vec<usize>,
    pub n_experts: usize,
    pub n_active: usize,
    /// Load balancing stats (after capacity truncation).
    pub load_stats: LoadStats,
}

impl DeviceGroupPlan {
    /// Original path (no capacity limit). Kept for compatibility.
    pub fn from_topk(
        topk_ids: &Tensor,
        topk_w: &Tensor,
        n_experts: usize,
        scale: f32,
    ) -> CandleResult<Self> {
        Self::from_topk_with_capacity(topk_ids, topk_w, n_experts, scale, 0.0)
    }

    /// Capacity-aware grouping (DeepSeek / Switch style).
    ///
    /// `capacity_factor <= 0.0` disables capacity (keeps all tokens).
    /// Otherwise:
    ///   capacity = max(1, ceil(capacity_factor * (t * k) / n_experts))
    /// For each expert we keep at most `capacity` highest-weight tokens and drop the rest.
    pub fn from_topk_with_capacity(
        topk_ids: &Tensor,
        topk_w: &Tensor,
        n_experts: usize,
        scale: f32,
        capacity_factor: f32,
    ) -> CandleResult<Self> {
        let (t, k) = topk_ids.dims2()?;
        let total = t * k;
        let device = topk_ids.device();

        let token_ids = Tensor::arange(0u32, t as u32, device)?
            .to_dtype(DType::I64)?
            .unsqueeze(1)?
            .expand((t, k))?
            .contiguous()?
            .flatten_all()?;

        let flat_experts = topk_ids.flatten_all()?.to_dtype(DType::I64)?;
        let flat_w = topk_w
            .flatten_all()?
            .to_dtype(DType::F32)?
            .affine(scale as f64, 0.0)?;

        // Pull to host for capacity truncation (offsets already host-side).
        let experts_v = flat_experts.to_vec1::<i64>()?;
        let tokens_v = token_ids.to_vec1::<i64>()?;
        let weights_v = flat_w.to_vec1::<f32>()?;

        // Group (token, weight) per expert, then keep top-capacity by weight.
        let mut per_expert: Vec<Vec<(i64, f32)>> = vec![Vec::new(); n_experts];
        for i in 0..total {
            let e = experts_v[i];
            if e >= 0 && (e as usize) < n_experts {
                per_expert[e as usize].push((tokens_v[i], weights_v[i]));
            }
        }

        let capacity = if capacity_factor <= 0.0 || n_experts == 0 {
            usize::MAX // no limit
        } else {
            let ideal = (capacity_factor * total as f32) / n_experts as f32;
            (ideal.ceil() as usize).max(1)
        };

        let mut kept_tokens: Vec<i64> = Vec::with_capacity(total);
        let mut kept_weights: Vec<f32> = Vec::with_capacity(total);
        let mut kept_experts: Vec<i64> = Vec::with_capacity(total);
        let mut tokens_per_expert = vec![0usize; n_experts];
        let mut dropped = 0usize;

        for (e, mut list) in per_expert.into_iter().enumerate() {
            if list.len() > capacity {
                // Keep highest weight (DeepSeek / Switch style)
                list.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
                    Some(o) => o,
                    None => Ordering::Less,
                });
                dropped += list.len() - capacity;
                list.truncate(capacity);
            }
            tokens_per_expert[e] = list.len();
            for (tok, w) in list {
                kept_tokens.push(tok);
                kept_weights.push(w);
                kept_experts.push(e as i64);
            }
        }

        let kept = kept_tokens.len();
        let max_load = tokens_per_expert.iter().copied().max().unwrap_or(0);
        let min_load = tokens_per_expert.iter().copied().min().unwrap_or(0);

        let load_stats = LoadStats {
            tokens_per_expert,
            total_assignments: total,
            kept_assignments: kept,
            dropped_assignments: dropped,
            capacity: if capacity == usize::MAX { 0 } else { capacity },
            max_load,
            min_load,
        };

        // Build final sorted tensors (already grouped by expert).
        let sorted_tokens = Tensor::from_vec(kept_tokens, (kept,), device)?;
        let sorted_weights = Tensor::from_vec(kept_weights, (kept,), device)?;
        let sorted_experts = Tensor::from_vec(kept_experts, (kept,), device)?;

        let expert_offsets = segment_offsets_fast(&sorted_experts, n_experts)?;

        if dropped > 0 {
            tracing::debug!(
                capacity = load_stats.capacity,
                dropped,
                drop_rate = %format!("{:.2}%", load_stats.drop_rate() * 100.0),
                max_load,
                min_load,
                imbalance = %format!("{:.2}", load_stats.imbalance_ratio()),
                cv = %format!("{:.3}", load_stats.coefficient_of_variation()),
                "MoE capacity drop applied"
            );
        }

        Ok(Self {
            sorted_tokens,
            sorted_weights,
            expert_offsets,
            n_experts,
            n_active: kept,
            load_stats,
        })
    }

    pub fn tokens_for(&self, e: usize) -> CandleResult<Option<Tensor>> {
        let (start, end) = (self.expert_offsets[e], self.expert_offsets[e + 1]);
        if start >= end {
            return Ok(None);
        }
        Ok(Some(
            self.sorted_tokens
                .narrow(0, start, end - start)?
                .to_dtype(DType::U32)?,
        ))
    }

    pub fn weights_for(&self, e: usize, dtype: DType) -> CandleResult<Option<Tensor>> {
        let (start, end) = (self.expert_offsets[e], self.expert_offsets[e + 1]);
        if start >= end {
            return Ok(None);
        }
        let w = self.sorted_weights.narrow(0, start, end - start)?;
        Ok(Some(w.to_dtype(dtype)?.unsqueeze(1)?))
    }

    pub fn occupancy(&self) -> &[usize] {
        &self.expert_offsets
    }

    pub fn load_stats(&self) -> &LoadStats {
        &self.load_stats
    }
}

fn segment_offsets_fast(sorted_experts: &Tensor, n_experts: usize) -> CandleResult<Vec<usize>> {
    let ids = sorted_experts.flatten_all()?.to_vec1::<i64>()?;
    let mut starts = vec![0usize; n_experts + 1];
    if ids.is_empty() {
        return Ok(starts);
    }
    let mut counts = vec![0usize; n_experts];
    for &e in &ids {
        if e >= 0 && (e as usize) < n_experts {
            counts[e as usize] += 1;
        }
    }
    for i in 0..n_experts {
        starts[i + 1] = starts[i] + counts[i];
    }
    Ok(starts)
}

/// Scatter-add — prefers Sentinel custom CUDA kernel (f32/f16/bf16).
pub fn device_scatter_add(
    base: &Tensor,
    token_indices: &Tensor,
    values: &Tensor,
) -> CandleResult<Tensor> {
    crate::kernels::cuda::scatter_add_rows(base, token_indices, values)
}

pub fn topk_last(x: &Tensor, k: usize) -> CandleResult<(Tensor, Tensor)> {
    let k = k.min(x.dim(D::Minus1)?);
    let order = x.to_dtype(DType::F32)?.contiguous()?.arg_sort_last_dim(false)?;
    let ids = order.narrow(D::Minus1, 0, k)?.contiguous()?;
    let vals = x.contiguous()?.gather(&ids, D::Minus1)?;
    Ok((vals, ids))
}
