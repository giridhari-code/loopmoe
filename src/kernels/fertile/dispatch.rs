//! Fertile MoE dispatch: route → group → expert → combine.

use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::Module;

#[derive(Clone, Copy, Debug)]
pub struct FertileMoeParams {
    pub num_experts_per_tok: usize,
    pub routed_scaling_factor: f32,
    pub norm_topk_prob: bool,
    pub use_sigmoid: bool,
}

pub trait FertileExpert {
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor>;
}

struct ExpertBuckets {
    token_indices: Vec<Vec<u32>>,
    weights: Vec<Vec<f32>>,
}

impl ExpertBuckets {
    fn from_topk(
        topk_ids: &[i64],
        topk_w: &[f32],
        num_tokens: usize,
        k: usize,
        n_routed: usize,
        scale: f32,
    ) -> Self {
        let mut token_indices = vec![Vec::new(); n_routed];
        let mut weights = vec![Vec::new(); n_routed];
        for tok in 0..num_tokens {
            for slot in 0..k {
                let idx = tok * k + slot;
                let e = topk_ids[idx] as usize;
                if e < n_routed {
                    token_indices[e].push(tok as u32);
                    weights[e].push(topk_w[idx] * scale);
                }
            }
        }
        Self { token_indices, weights }
    }
}

pub struct FertileDispatch;

impl FertileDispatch {
    pub fn forward<E: FertileExpert>(
        x_flat: &Tensor,
        gate: &impl Module,
        routed: &[E],
        shared: &[E],
        params: FertileMoeParams,
    ) -> CandleResult<Tensor> {
        let (t, h) = x_flat.dims2()?;
        let device = x_flat.device();
        let dtype = x_flat.dtype();

        let mut out = Tensor::zeros((t, h), dtype, device)?;
        for expert in shared {
            out = (out + expert.forward(x_flat)?)?;
        }

        if routed.is_empty() || params.num_experts_per_tok == 0 {
            return Ok(out);
        }

        let logits = gate.forward(x_flat)?;
        let scores = if params.use_sigmoid {
            candle_nn::ops::sigmoid(&logits)?
        } else {
            logits
        };

        let (mut topk_w, topk_ids) = scores.topk(params.num_experts_per_tok)?;
        let topk_ids = topk_ids.to_dtype(DType::I64)?;

        topk_w = if params.use_sigmoid {
            if params.norm_topk_prob {
                let sum = topk_w.sum_keepdim(1)?;
                topk_w.broadcast_div(&(sum + 1e-20)?)?
            } else {
                topk_w
            }
        } else {
            candle_nn::ops::softmax_last_dim(&topk_w)?
        };

        let ids_v = topk_ids.flatten_all()?.to_vec1::<i64>()?;
        let w_v = topk_w.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let buckets = ExpertBuckets::from_topk(
            &ids_v,
            &w_v,
            t,
            params.num_experts_per_tok,
            routed.len(),
            params.routed_scaling_factor,
        );

        let mut routed_acc = Tensor::zeros((t, h), dtype, device)?;
        for (e, (toks, ws)) in buckets
            .token_indices
            .into_iter()
            .zip(buckets.weights)
            .enumerate()
        {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let indices = Tensor::from_vec(toks, (n,), device)?;
            let w = Tensor::from_vec(ws, (n, 1), device)?.to_dtype(dtype)?;
            let selected = x_flat.index_select(&indices, 0)?;
            let y = routed[e].forward(&selected)?;
            let y = y.broadcast_mul(&w)?;
            routed_acc = scatter_add(&routed_acc, &indices, &y)?;
        }

        out + routed_acc
    }
}

pub fn fertile_moe<E: FertileExpert>(
    x_flat: &Tensor,
    gate: &impl Module,
    routed: &[E],
    shared: &[E],
    params: FertileMoeParams,
) -> CandleResult<Tensor> {
    FertileDispatch::forward(x_flat, gate, routed, shared, params)
}

fn scatter_add(base: &Tensor, indices: &Tensor, values: &Tensor) -> CandleResult<Tensor> {
    let h = values.dim(1)?;
    let n = indices.dims1()?;
    let indices_exp = indices.unsqueeze(1)?.expand((n, h))?.contiguous()?;
    base.scatter_add(&indices_exp, values, 0)
}
