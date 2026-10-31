//! Parallel configuration.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelismKind {
    /// Single process, single GPU (or CPU).
    Single,
    /// Tensor Parallel (column/row split of linear layers).
    Tensor,
    /// Expert Parallel (experts sharded across ranks — DeepEP style).
    Expert,
    /// Data Parallel (replica + gradient all-reduce).
    Data,
    /// Hybrid TP × EP × DP.
    Hybrid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelConfig {
    /// Total processes in the job.
    pub world_size: usize,
    /// This process rank [0, world_size).
    pub rank: usize,
    /// Local GPU index on the node.
    pub local_rank: usize,
    /// Tensor-parallel degree.
    pub tp_size: usize,
    /// Expert-parallel degree (DeepEP-style).
    pub ep_size: usize,
    /// Data-parallel degree.
    pub dp_size: usize,
    /// Prefer expert parallel when MoE is active.
    pub prefer_ep: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            local_rank: 0,
            tp_size: 1,
            ep_size: 1,
            dp_size: 1,
            prefer_ep: true,
        }
    }
}

impl ParallelConfig {
    pub fn single() -> Self {
        Self::default()
    }

    pub fn is_single(&self) -> bool {
        self.world_size == 1 && self.tp_size == 1 && self.ep_size == 1 && self.dp_size == 1
    }

    pub fn kind(&self) -> ParallelismKind {
        if self.is_single() {
            return ParallelismKind::Single;
        }
        let multi_tp = self.tp_size > 1;
        let multi_ep = self.ep_size > 1;
        let multi_dp = self.dp_size > 1;
        match (multi_tp, multi_ep, multi_dp) {
            (false, true, false) => ParallelismKind::Expert,
            (true, false, false) => ParallelismKind::Tensor,
            (false, false, true) => ParallelismKind::Data,
            _ => ParallelismKind::Hybrid,
        }
    }

    /// Validate that tp × ep × dp == world_size (when multi-GPU).
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.world_size == 0 {
            anyhow::bail!("world_size must be >= 1");
        }
        if self.rank >= self.world_size {
            anyhow::bail!("rank {} >= world_size {}", self.rank, self.world_size);
        }
        let product = self.tp_size.saturating_mul(self.ep_size).saturating_mul(self.dp_size);
        if self.world_size > 1 && product != self.world_size {
            anyhow::bail!(
                "tp({}) × ep({}) × dp({}) = {} != world_size({})",
                self.tp_size,
                self.ep_size,
                self.dp_size,
                product,
                self.world_size
            );
        }
        Ok(())
    }

    /// Build from environment (TORCH / RANK style) when available.
    pub fn from_env() -> Self {
        let world_size = std::env::var("WORLD_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let rank = std::env::var("RANK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let local_rank = std::env::var("LOCAL_RANK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let ep_size = std::env::var("EP_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(if world_size > 1 { world_size } else { 1 });
        let tp_size = std::env::var("TP_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let dp_size = if tp_size * ep_size > 0 && world_size % (tp_size * ep_size) == 0 {
            world_size / (tp_size * ep_size).max(1)
        } else {
            1
        };
        Self {
            world_size,
            rank,
            local_rank,
            tp_size,
            ep_size,
            dp_size,
            prefer_ep: true,
        }
    }
}
