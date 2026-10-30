//! KV Cache for autoregressive generation.
//! Phase 1: contiguous cache per physical layer.
//! Future: PagedAttention (DeepInfer block_manager style).

use candle_core::{Result as CandleResult, Tensor};

#[derive(Debug, Clone)]
pub struct KvCache {
    /// [batch, num_kv_heads, seq, head_dim]
    pub k: Option<Tensor>,
    pub v: Option<Tensor>,
}

impl KvCache {
    pub fn empty() -> Self {
        Self { k: None, v: None }
    }

    pub fn seq_len(&self) -> usize {
        match &self.k {
            Some(k) => k.dim(2).unwrap_or(0),
            None => 0,
        }
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> CandleResult<()> {
        self.k = Some(match &self.k {
            None => new_k.clone(),
            Some(prev) => Tensor::cat(&[prev, new_k], 2)?,
        });
        self.v = Some(match &self.v {
            None => new_v.clone(),
            Some(prev) => Tensor::cat(&[prev, new_v], 2)?,
        });
        Ok(())
    }

    pub fn clear(&mut self) {
        self.k = None;
        self.v = None;
    }
}

pub struct LayerCaches {
    pub layers: Vec<KvCache>,
}

impl LayerCaches {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| KvCache::empty()).collect(),
        }
    }

    pub fn clear(&mut self) {
        for c in &mut self.layers {
            c.clear();
        }
    }

    pub fn current_len(&self) -> usize {
        self.layers.first().map(|c| c.seq_len()).unwrap_or(0)
    }
}
