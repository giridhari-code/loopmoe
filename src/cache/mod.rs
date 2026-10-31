//! KV / MLA cache — preallocated capacity to avoid O(n²) Tensor::cat growth.

use candle_core::{DType, Device, Result as CandleResult, Tensor};

// ---------------------------------------------------------------------------
// GQA KV cache
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct KvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
    capacity: usize,
    batch: usize,
    num_kv_heads: usize,
    head_dim: usize,
    dtype: DType,
    device: Device,
}

impl KvCache {
    pub fn with_capacity(
        batch: usize,
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
            capacity: capacity.max(1),
            batch,
            num_kv_heads,
            head_dim,
            dtype,
            device: device.clone(),
        }
    }

    pub fn empty() -> Self {
        Self {
            k: None,
            v: None,
            len: 0,
            capacity: 0,
            batch: 1,
            num_kv_heads: 0,
            head_dim: 0,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    pub fn seq_len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_allocated(&mut self) -> CandleResult<()> {
        if self.k.is_some() {
            return Ok(());
        }
        let shape = (self.batch, self.num_kv_heads, self.capacity, self.head_dim);
        self.k = Some(Tensor::zeros(shape, self.dtype, &self.device)?);
        self.v = Some(Tensor::zeros(shape, self.dtype, &self.device)?);
        Ok(())
    }

    fn grow_to(&mut self, need: usize) -> CandleResult<()> {
        if self.k.is_none() {
            if need > self.capacity {
                self.capacity = need;
            }
            return Ok(());
        }
        if need <= self.capacity {
            return Ok(());
        }
        let new_cap = need.max(self.capacity.max(1) * 2).max(need);
        let shape = (self.batch, self.num_kv_heads, new_cap, self.head_dim);
        let mut new_k = Tensor::zeros(shape, self.dtype, &self.device)?;
        let mut new_v = Tensor::zeros(shape, self.dtype, &self.device)?;
        if let (Some(ok), Some(ov)) = (&self.k, &self.v) {
            if self.len > 0 {
                let old_k = ok.narrow(2, 0, self.len)?;
                let old_v = ov.narrow(2, 0, self.len)?;
                let tail_k = new_k.narrow(2, self.len, new_cap - self.len)?;
                let tail_v = new_v.narrow(2, self.len, new_cap - self.len)?;
                new_k = Tensor::cat(&[&old_k, &tail_k], 2)?;
                new_v = Tensor::cat(&[&old_v, &tail_v], 2)?;
            }
        }
        self.k = Some(new_k);
        self.v = Some(new_v);
        self.capacity = new_cap;
        Ok(())
    }

    pub fn append(&mut self, new_k: &Tensor, new_v: &Tensor) -> CandleResult<()> {
        let s_new = new_k.dim(2)?;
        let need = self.len + s_new;
        if self.capacity == 0 {
            self.batch = new_k.dim(0)?;
            self.num_kv_heads = new_k.dim(1)?;
            self.head_dim = new_k.dim(3)?;
            self.dtype = new_k.dtype();
            self.device = new_k.device().clone();
            self.capacity = need.max(64);
        }
        self.grow_to(need)?;
        self.ensure_allocated()?;

        let start = self.len;
        let k = self.k.as_ref().unwrap();
        let v = self.v.as_ref().unwrap();
        k.slice_set(&new_k.contiguous()?, 2, start)?;
        v.slice_set(&new_v.contiguous()?, 2, start)?;
        self.len = start + s_new;
        Ok(())
    }

    pub fn k_view(&self) -> CandleResult<Option<Tensor>> {
        if self.len == 0 {
            return Ok(None);
        }
        Ok(Some(self.k.as_ref().unwrap().narrow(2, 0, self.len)?))
    }

    pub fn v_view(&self) -> CandleResult<Option<Tensor>> {
        if self.len == 0 {
            return Ok(None);
        }
        Ok(Some(self.v.as_ref().unwrap().narrow(2, 0, self.len)?))
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn set_len(&mut self, len: usize) {
        self.len = len.min(self.capacity);
    }

    pub fn release(&mut self) {
        self.k = None;
        self.v = None;
        self.len = 0;
    }
}

#[derive(Debug)]
pub struct LayerCaches {
    pub layers: Vec<KvCache>,
    pub capacity: usize,
}

impl LayerCaches {
    pub fn with_capacity(
        num_layers: usize,
        batch: usize,
        num_kv_heads: usize,
        head_dim: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let capacity = capacity.max(1);
        Self {
            layers: (0..num_layers)
                .map(|_| {
                    KvCache::with_capacity(batch, num_kv_heads, head_dim, capacity, dtype, device)
                })
                .collect(),
            capacity,
        }
    }

    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| KvCache::empty()).collect(),
            capacity: 0,
        }
    }

    pub fn clear(&mut self) {
        for c in &mut self.layers {
            c.clear();
        }
    }

    pub fn release(&mut self) {
        for c in &mut self.layers {
            c.release();
        }
    }

    pub fn current_len(&self) -> usize {
        self.layers.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    pub fn checkpoint_lens(&self) -> Vec<usize> {
        self.layers.iter().map(|c| c.seq_len()).collect()
    }

    pub fn restore_lens(&mut self, lens: &[usize]) {
        for (c, &l) in self.layers.iter_mut().zip(lens) {
            c.set_len(l);
        }
    }
}

pub fn kv_bytes_estimate(
    num_layers: usize,
    batch: usize,
    num_kv_heads: usize,
    head_dim: usize,
    seq: usize,
    bytes_per_elem: usize,
) -> usize {
    num_layers * 2 * batch * num_kv_heads * head_dim * seq * bytes_per_elem
}

// ---------------------------------------------------------------------------
// MLA cache (latent c_kv + rope key) — same RAM strategy as GQA
// c:    (batch, capacity, rank)
// k_pe: (batch, 1, capacity, rope_dim)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct MlaCache {
    c: Option<Tensor>,
    k_pe: Option<Tensor>,
    len: usize,
    capacity: usize,
    batch: usize,
    rank: usize,
    rope_dim: usize,
    dtype: DType,
    device: Device,
}

impl MlaCache {
    pub fn empty() -> Self {
        Self {
            c: None,
            k_pe: None,
            len: 0,
            capacity: 0,
            batch: 1,
            rank: 0,
            rope_dim: 0,
            dtype: DType::F32,
            device: Device::Cpu,
        }
    }

    /// Preallocate for `capacity` sequence positions.
    pub fn with_capacity(
        batch: usize,
        rank: usize,
        rope_dim: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Self {
        Self {
            c: None,
            k_pe: None,
            len: 0,
            capacity: capacity.max(1),
            batch,
            rank,
            rope_dim,
            dtype,
            device: device.clone(),
        }
    }

    pub fn seq_len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    fn ensure_allocated(&mut self) -> CandleResult<()> {
        if self.c.is_some() {
            return Ok(());
        }
        let c_shape = (self.batch, self.capacity, self.rank);
        let k_shape = (self.batch, 1, self.capacity, self.rope_dim);
        self.c = Some(Tensor::zeros(c_shape, self.dtype, &self.device)?);
        self.k_pe = Some(Tensor::zeros(k_shape, self.dtype, &self.device)?);
        Ok(())
    }

    fn grow_to(&mut self, need: usize) -> CandleResult<()> {
        if self.c.is_none() {
            if need > self.capacity {
                self.capacity = need;
            }
            return Ok(());
        }
        if need <= self.capacity {
            return Ok(());
        }
        let new_cap = need.max(self.capacity.max(1) * 2).max(need);
        let mut new_c = Tensor::zeros((self.batch, new_cap, self.rank), self.dtype, &self.device)?;
        let mut new_k =
            Tensor::zeros((self.batch, 1, new_cap, self.rope_dim), self.dtype, &self.device)?;
        if let (Some(oc), Some(ok)) = (&self.c, &self.k_pe) {
            if self.len > 0 {
                let old_c = oc.narrow(1, 0, self.len)?;
                let old_k = ok.narrow(2, 0, self.len)?;
                let tail_c = new_c.narrow(1, self.len, new_cap - self.len)?;
                let tail_k = new_k.narrow(2, self.len, new_cap - self.len)?;
                new_c = Tensor::cat(&[&old_c, &tail_c], 1)?;
                new_k = Tensor::cat(&[&old_k, &tail_k], 2)?;
            }
        }
        self.c = Some(new_c);
        self.k_pe = Some(new_k);
        self.capacity = new_cap;
        Ok(())
    }

    /// In-place append — no O(n²) cat on the hot path.
    /// `c_kv`: (B, S, rank)   `k_pe`: (B, 1, S, rope_dim)
    pub fn append(&mut self, c_kv: &Tensor, k_pe: &Tensor) -> CandleResult<()> {
        let s_new = c_kv.dim(1)?;
        let need = self.len + s_new;

        if self.capacity == 0 || self.rank == 0 {
            self.batch = c_kv.dim(0)?;
            self.rank = c_kv.dim(2)?;
            self.rope_dim = k_pe.dim(3)?;
            self.dtype = c_kv.dtype();
            self.device = c_kv.device().clone();
            self.capacity = need.max(64);
        }

        self.grow_to(need)?;
        self.ensure_allocated()?;

        let start = self.len;
        let c = self.c.as_ref().unwrap();
        let k = self.k_pe.as_ref().unwrap();
        c.slice_set(&c_kv.contiguous()?, 1, start)?;
        k.slice_set(&k_pe.contiguous()?, 2, start)?;
        self.len = start + s_new;
        Ok(())
    }

    pub fn c_view(&self) -> CandleResult<Option<Tensor>> {
        if self.len == 0 {
            return Ok(None);
        }
        Ok(Some(self.c.as_ref().unwrap().narrow(1, 0, self.len)?))
    }

    pub fn k_pe_view(&self) -> CandleResult<Option<Tensor>> {
        if self.len == 0 {
            return Ok(None);
        }
        Ok(Some(self.k_pe.as_ref().unwrap().narrow(2, 0, self.len)?))
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn set_len(&mut self, len: usize) {
        self.len = len.min(if self.capacity > 0 {
            self.capacity
        } else {
            self.len
        });
    }

    pub fn release(&mut self) {
        self.c = None;
        self.k_pe = None;
        self.len = 0;
    }
}

impl Default for MlaCache {
    fn default() -> Self {
        Self::empty()
    }
}

#[derive(Debug)]
pub struct MlaCaches {
    pub layers: Vec<MlaCache>,
    pub capacity: usize,
}

impl MlaCaches {
    /// Lazy empty layers (no prealloc) — prefer `with_capacity`.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| MlaCache::empty()).collect(),
            capacity: 0,
        }
    }

    /// Preallocate all layers for `capacity` tokens.
    pub fn with_capacity(
        num_layers: usize,
        batch: usize,
        kv_lora_rank: usize,
        rope_dim: usize,
        capacity: usize,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let capacity = capacity.max(1);
        Self {
            layers: (0..num_layers)
                .map(|_| {
                    MlaCache::with_capacity(
                        batch,
                        kv_lora_rank,
                        rope_dim,
                        capacity,
                        dtype,
                        device,
                    )
                })
                .collect(),
            capacity,
        }
    }

    pub fn clear(&mut self) {
        for c in &mut self.layers {
            c.clear();
        }
    }

    pub fn release(&mut self) {
        for c in &mut self.layers {
            c.release();
        }
    }

    pub fn current_len(&self) -> usize {
        self.layers.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    pub fn checkpoint_lens(&self) -> Vec<usize> {
        self.layers.iter().map(|c| c.seq_len()).collect()
    }

    pub fn restore_lens(&mut self, lens: &[usize]) {
        for (c, &l) in self.layers.iter_mut().zip(lens) {
            c.set_len(l);
        }
    }
}

/// Bytes for MLA latent cache: c_kv + k_pe per layer.
pub fn mla_bytes_estimate(
    num_layers: usize,
    batch: usize,
    kv_lora_rank: usize,
    rope_dim: usize,
    seq: usize,
    bytes_per_elem: usize,
) -> usize {
    // c: (B, S, rank) + k_pe: (B, 1, S, rope)
    let per_layer = batch * seq * kv_lora_rank + batch * seq * rope_dim;
    num_layers * per_layer * bytes_per_elem
}

// ---------------------------------------------------------------------------
// Unified runtime bank
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum RuntimeCaches {
    Gqa(LayerCaches),
    Mla(MlaCaches),
}

impl RuntimeCaches {
    pub fn current_len(&self) -> usize {
        match self {
            RuntimeCaches::Gqa(b) => b.current_len(),
            RuntimeCaches::Mla(b) => b.current_len(),
        }
    }

    pub fn clear(&mut self) {
        match self {
            RuntimeCaches::Gqa(b) => b.clear(),
            RuntimeCaches::Mla(b) => b.clear(),
        }
    }

    pub fn release(&mut self) {
        match self {
            RuntimeCaches::Gqa(b) => b.release(),
            RuntimeCaches::Mla(b) => b.release(),
        }
    }

    pub fn checkpoint_lens(&self) -> Vec<usize> {
        match self {
            RuntimeCaches::Gqa(b) => b.checkpoint_lens(),
            RuntimeCaches::Mla(b) => b.checkpoint_lens(),
        }
    }

    pub fn restore_lens(&mut self, lens: &[usize]) {
        match self {
            RuntimeCaches::Gqa(b) => b.restore_lens(lens),
            RuntimeCaches::Mla(b) => b.restore_lens(lens),
        }
    }
}
