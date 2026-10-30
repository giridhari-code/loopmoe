//! KV cache — preallocated capacity to avoid O(n²) Tensor::cat growth.

use candle_core::{DType, Device, Result as CandleResult, Tensor};

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
        if need <= self.capacity && self.k.is_some() {
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

        let k = self.k.as_mut().unwrap();
        let v = self.v.as_mut().unwrap();
        let start = self.len;
        let end = start + s_new;
        *k = slice_assign_seq(k, start, end, new_k)?;
        *v = slice_assign_seq(v, start, end, new_v)?;
        self.len = end;
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

    pub fn release(&mut self) {
        self.k = None;
        self.v = None;
        self.len = 0;
    }
}

fn slice_assign_seq(buf: &Tensor, start: usize, end: usize, src: &Tensor) -> CandleResult<Tensor> {
    let (_b, _h, cap, _d) = buf.dims4()?;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if start > 0 {
        parts.push(buf.narrow(2, 0, start)?);
    }
    parts.push(src.clone());
    if end < cap {
        parts.push(buf.narrow(2, end, cap - end)?);
    }
    Tensor::cat(&parts, 2)
}

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
