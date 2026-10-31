//! Device mesh — maps (tp, ep, dp) ranks to physical devices.

use candle_core::{Device, Result as CandleResult};
use super::config::ParallelConfig;

#[derive(Debug, Clone)]
pub struct RankInfo {
    pub global_rank: usize,
    pub local_rank: usize,
    pub tp_rank: usize,
    pub ep_rank: usize,
    pub dp_rank: usize,
}

#[derive(Debug, Clone)]
pub struct DeviceMesh {
    pub config: ParallelConfig,
    pub device: Device,
    pub rank_info: RankInfo,
}

impl DeviceMesh {
    /// Single-GPU / single-process mesh (default path).
    pub fn single(device: Device) -> Self {
        Self {
            config: ParallelConfig::single(),
            device,
            rank_info: RankInfo {
                global_rank: 0,
                local_rank: 0,
                tp_rank: 0,
                ep_rank: 0,
                dp_rank: 0,
            },
        }
    }

    pub fn from_config(cfg: ParallelConfig, device: Device) -> anyhow::Result<Self> {
        cfg.validate()?;
        let tp = cfg.tp_size.max(1);
        let ep = cfg.ep_size.max(1);
        // Rank layout: [dp][ep][tp]  (tp is fastest-changing)
        let tp_rank = cfg.rank % tp;
        let ep_rank = (cfg.rank / tp) % ep;
        let dp_rank = cfg.rank / (tp * ep);
        Ok(Self {
            rank_info: RankInfo {
                global_rank: cfg.rank,
                local_rank: cfg.local_rank,
                tp_rank,
                ep_rank,
                dp_rank,
            },
            config: cfg,
            device,
        })
    }

    pub fn is_single(&self) -> bool {
        self.config.is_single()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Select CUDA device by local_rank when available.
    pub fn select_device(local_rank: usize) -> CandleResult<Device> {
        if let Ok(d) = Device::new_cuda(local_rank) {
            return Ok(d);
        }
        // Fallback: try cuda:0 then CPU
        if let Ok(d) = Device::new_cuda(0) {
            return Ok(d);
        }
        Ok(Device::Cpu)
    }
}
