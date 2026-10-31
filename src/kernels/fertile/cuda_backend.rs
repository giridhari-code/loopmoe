//! Fertile CUDA backend — device grouping + **parallel** expert GEMMs.

use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::Module;
use rayon::prelude::*;

use super::dispatch::{FertileExpert, FertileMoeParams};
use super::group_plan::{device_scatter_add, topk_last, DeviceGroupPlan};

pub fn fertile_moe_cuda<E: FertileExpert + Sync>(
    x_flat: &Tensor,
    gate: &impl Module,
    routed: &[E],
    shared: &[E],
    params: FertileMoeParams,
) -> CandleResult<Tensor> {
    debug_assert!(x_flat.device().is_cuda(), "fertile_moe_cuda requires CUDA tensor");

    let (t, h) = x_flat.dims2()?;
    let device = x_flat.device();
    let dtype = x_flat.dtype();

    // Shared experts — accumulate (usually few; keep sequential add for stability)
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

    let plan = DeviceGroupPlan::from_topk(
        &topk_ids,
        &topk_w,
        routed.len(),
        params.routed_scaling_factor,
    )?;

    // Collect active experts first (parallel-friendly work list)
    let active: Vec<usize> = (0..plan.n_experts)
        .filter(|&e| {
            let (s, end) = (plan.expert_offsets[e], plan.expert_offsets[e + 1]);
            end > s
        })
        .collect();

    // Parallel map: each active expert runs its own gather → GEMM → weighted out
    // Results are independent; we scatter-add sequentially only for the final reduce
    // (scatter_add on same buffer is not data-race safe across threads).
    let partials: Vec<CandleResult<(Tensor, Tensor)>> = active
        .par_iter()
        .map(|&e| {
            let indices = plan.tokens_for(e)?.ok_or_else(|| {
                candle_core::Error::Msg(format!("missing indices expert {e}"))
            })?;
            let w = plan.weights_for(e, dtype)?.ok_or_else(|| {
                candle_core::Error::Msg(format!("missing weights expert {e}"))
            })?;
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
