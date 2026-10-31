//! Device-side token→expert grouping for Fertile.

use candle_core::{DType, Result as CandleResult, Tensor, D};

pub struct DeviceGroupPlan {
    pub sorted_tokens: Tensor,
    pub sorted_weights: Tensor,
    pub expert_offsets: Vec<usize>,
    pub n_experts: usize,
    pub n_active: usize,
}

impl DeviceGroupPlan {
    pub fn from_topk(
        topk_ids: &Tensor,
        topk_w: &Tensor,
        n_experts: usize,
        scale: f32,
    ) -> CandleResult<Self> {
        let (t, k) = topk_ids.dims2()?;
        let n = t * k;
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

        let order = argsort_i64(&flat_experts)?;
        let sorted_experts = flat_experts.gather(&order, 0)?;
        let sorted_tokens = token_ids.gather(&order, 0)?;
        let sorted_weights = flat_w.gather(&order, 0)?;

        let expert_offsets = segment_offsets(&sorted_experts, n_experts)?;

        Ok(Self {
            sorted_tokens,
            sorted_weights,
            expert_offsets,
            n_experts,
            n_active: n,
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
}

fn argsort_i64(x: &Tensor) -> CandleResult<Tensor> {
    let x2 = if x.rank() == 1 {
        x.unsqueeze(0)?
    } else {
        x.clone()
    };
    match x2.arg_sort_last_dim(true) {
        Ok(order) => Ok(order.flatten_all()?),
        Err(_) => {
            let v = x.flatten_all()?.to_vec1::<i64>()?;
            let mut idx: Vec<i64> = (0..v.len() as i64).collect();
            idx.sort_by_key(|&i| v[i as usize]);
            Tensor::from_vec(idx, (v.len(),), x.device())
        }
    }
}

fn segment_offsets(sorted_experts: &Tensor, n_experts: usize) -> CandleResult<Vec<usize>> {
    let ids = sorted_experts.flatten_all()?.to_vec1::<i64>()?;
    let mut counts = vec![0usize; n_experts];
    for &e in &ids {
        if (0..n_experts as i64).contains(&e) {
            counts[e as usize] += 1;
        }
    }
    let mut starts = vec![0usize; n_experts + 1];
    for i in 0..n_experts {
        starts[i + 1] = starts[i] + counts[i];
    }
    Ok(starts)
}

pub fn device_scatter_add(
    base: &Tensor,
    token_indices: &Tensor,
    values: &Tensor,
) -> CandleResult<Tensor> {
    let h = values.dim(1)?;
    let n = token_indices.dims1()?;
    let idx = token_indices
        .to_dtype(DType::U32)?
        .unsqueeze(1)?
        .expand((n, h))?
        .contiguous()?;
    base.scatter_add(&idx, values, 0)
}

/// Top-k along the last dim (candle-core 0.8 has no `Tensor::topk`).
/// Returns (values, indices[u32]) sorted by descending value.
pub fn topk_last(x: &Tensor, k: usize) -> CandleResult<(Tensor, Tensor)> {
    let order = x.to_dtype(DType::F32)?.contiguous()?.arg_sort_last_dim(false)?;
    let ids = order.narrow(D::Minus1, 0, k)?.contiguous()?;
    let vals = x.contiguous()?.gather(&ids, D::Minus1)?;
    Ok((vals, ids))
}
