//! Fertile ARM/CPU backend — same plan, **parallel** expert map.

use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::Module;
use rayon::prelude::*;

use super::dispatch::{FertileExpert, FertileMoeParams};
use super::group_plan::{device_scatter_add, topk_last, DeviceGroupPlan};

pub fn fertile_moe_arm<E: FertileExpert + Sync>(
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

    let (mut topk_w, topk_ids) = topk_last(&scores, params.num_experts_per_tok)?;
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

    let plan = DeviceGroupPlan::from_topk_with_capacity(
        &topk_ids,
        &topk_w,
        routed.len(),
        params.routed_scaling_factor,
        params.capacity_factor,
    )?;

    let active: Vec<usize> = (0..plan.n_experts)
        .filter(|&e| plan.expert_offsets[e + 1] > plan.expert_offsets[e])
        .collect();

    let partials: Vec<CandleResult<(Tensor, Tensor)>> = active
        .par_iter()
        .map(|&e| {
            let indices = plan
                .tokens_for(e)?
                .ok_or_else(|| candle_core::Error::Msg(format!("missing indices {e}")))?;
            let w = plan
                .weights_for(e, dtype)?
                .ok_or_else(|| candle_core::Error::Msg(format!("missing weights {e}")))?;
            let selected = x_flat.index_select(&indices, 0)?;
            let y = routed[e].forward(&selected)?.broadcast_mul(&w)?;
            Ok((indices, y))
        })
        .collect();

    let mut routed_acc = Tensor::zeros((t, h), dtype, device)?;
    for p in partials {
        let (indices, y) = p?;
        routed_acc = device_scatter_add(&routed_acc, &indices, &y)?;
    }

    out + routed_acc
}
