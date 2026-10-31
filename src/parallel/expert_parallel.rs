//! Expert Parallel plan — DeepEP-inspired sharding.
//!
//! Experts are partitioned across `ep_size` ranks. Each rank owns a contiguous
//! shard of routed experts and runs local MoE on tokens dispatched to it.

use super::config::ParallelConfig;
use super::mesh::DeviceMesh;

#[derive(Debug, Clone)]
pub struct ExpertShard {
    /// Global expert id range [start, end).
    pub start: usize,
    pub end: usize,
    /// Number of experts on this rank.
    pub count: usize,
    pub ep_rank: usize,
    pub ep_size: usize,
}

impl ExpertShard {
    pub fn contains(&self, expert_id: usize) -> bool {
        expert_id >= self.start && expert_id < self.end
    }

    pub fn local_id(&self, global_id: usize) -> Option<usize> {
        if self.contains(global_id) {
            Some(global_id - self.start)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExpertParallelPlan {
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub shard: ExpertShard,
    /// Shared experts are typically replicated on every rank.
    pub shared_replicated: bool,
}

impl ExpertParallelPlan {
    /// Build plan for single-GPU (owns all experts).
    pub fn single(n_routed: usize, n_shared: usize, top_k: usize) -> Self {
        Self {
            n_routed_experts: n_routed,
            n_shared_experts: n_shared,
            num_experts_per_tok: top_k,
            shard: ExpertShard {
                start: 0,
                end: n_routed,
                count: n_routed,
                ep_rank: 0,
                ep_size: 1,
            },
            shared_replicated: true,
        }
    }

    pub fn from_mesh(
        mesh: &DeviceMesh,
        n_routed: usize,
        n_shared: usize,
        top_k: usize,
    ) -> anyhow::Result<Self> {
        let ep = mesh.config.ep_size.max(1);
        let ep_rank = mesh.rank_info.ep_rank;
        if n_routed % ep != 0 {
            // Uneven last shard allowed; prefer even for balance.
            tracing::warn!(
                n_routed,
                ep,
                "n_routed_experts not divisible by ep_size — last rank gets remainder"
            );
        }
        let base = n_routed / ep;
        let rem = n_routed % ep;
        let start = ep_rank * base + ep_rank.min(rem);
        let count = base + if ep_rank < rem { 1 } else { 0 };
        let end = start + count;
        Ok(Self {
            n_routed_experts: n_routed,
            n_shared_experts: n_shared,
            num_experts_per_tok: top_k,
            shard: ExpertShard {
                start,
                end,
                count,
                ep_rank,
                ep_size: ep,
            },
            shared_replicated: true,
        })
    }

    pub fn owns_all(&self) -> bool {
        self.shard.ep_size == 1
    }

    pub fn local_routed_count(&self) -> usize {
        self.shard.count
    }
}

/// Helper: which global experts should be dispatched to `ep_rank`.
pub fn experts_for_rank(n_routed: usize, ep_size: usize, ep_rank: usize) -> (usize, usize) {
    let base = n_routed / ep_size.max(1);
    let rem = n_routed % ep_size.max(1);
    let start = ep_rank * base + ep_rank.min(rem);
    let count = base + if ep_rank < rem { 1 } else { 0 };
    (start, start + count)
}

impl From<&ParallelConfig> for Option<ExpertParallelPlan> {
    fn from(_cfg: &ParallelConfig) -> Self {
        None // filled by caller with expert counts
    }
}
