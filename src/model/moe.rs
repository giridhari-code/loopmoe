use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use crate::config::ModelConfig;

/// Single expert (SwiGLU style: silu(w1(x)) * w3(x) → w2)
pub struct Expert {
    w1: Linear, // gate
    w2: Linear, // down
    w3: Linear, // up
}

impl Expert {
    pub fn load(vb: VarBuilder, hidden: usize, intermediate: usize) -> CandleResult<Self> {
        Ok(Self {
            w1: linear_no_bias(hidden, intermediate, vb.pp("w1"))?,
            w2: linear_no_bias(intermediate, hidden, vb.pp("w2"))?,
            w3: linear_no_bias(hidden, intermediate, vb.pp("w3"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

/// Sparse Mixture-of-Experts layer with top-k routing.
///
/// Phase 1: correct top-k + weighted sum of expert outputs.
/// Later we will replace the expert loop with grouped GEMM / DeepEP-style
/// dispatch inspired by DeepInfer kernels.
pub struct MoeLayer {
    gate: Linear,
    experts: Vec<Expert>,
    num_experts_per_tok: usize,
    num_experts: usize,
    norm_topk_prob: bool,
}

impl MoeLayer {
    pub fn load(vb: VarBuilder, config: &ModelConfig) -> CandleResult<Self> {
        let gate = linear_no_bias(config.hidden_size, config.num_experts, vb.pp("gate"))?;

        let mut experts = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            experts.push(Expert::load(
                vb.pp(&format!("experts.{i}")),
                config.hidden_size,
                config.moe_intermediate_size,
            )?);
        }

        Ok(Self {
            gate,
            experts,
            num_experts_per_tok: config.num_experts_per_tok,
            num_experts: config.num_experts,
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    /// Forward with top-k routing.
    ///
    /// x: [batch, seq, hidden]
    /// returns: [batch, seq, hidden]
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, s, h) = x.dims3()?;
        let num_tokens = b * s;
        let x_flat = x.reshape((num_tokens, h))?; // [T, H]

        // 1. Router logits
        let router_logits = self.gate.forward(&x_flat)?; // [T, E]

        // 2. Top-k
        let (topk_weight, topk_ids) = router_logits.topk(self.num_experts_per_tok)?;
        // topk_ids is typically I64

        // 3. Softmax over the selected experts
        let topk_weight = candle_nn::ops::softmax_last_dim(&topk_weight)?;

        // 4. Weighted sum of expert outputs
        let mut final_hidden = Tensor::zeros((num_tokens, h), x.dtype(), x.device())?;

        // Convert ids to i64 for consistent comparison
        let topk_ids = topk_ids.to_dtype(DType::I64)?;

        for k in 0..self.num_experts_per_tok {
            let expert_ids_k = topk_ids.i((.., k))?; // [T] i64
            let weights_k = topk_weight.i((.., k))?; // [T]

            for expert_idx in 0..self.num_experts {
                let expert_id_val =
                    Tensor::new(expert_idx as i64, expert_ids_k.device())?
                        .broadcast_as(expert_ids_k.shape())?;
                let mask = expert_ids_k.eq(&expert_id_val)?;

                let count = mask.to_dtype(DType::U32)?.sum_all()?.to_scalar::<u32>()?;
                if count == 0 {
                    continue;
                }

                let indices = compact_indices(&mask)?;
                if indices.dims1()? == 0 {
                    continue;
                }

                let selected_x = x_flat.index_select(&indices, 0)?;
                let selected_w = weights_k.index_select(&indices, 0)?.unsqueeze(1)?;

                let expert_out = self.experts[expert_idx].forward(&selected_x)?;
                let weighted = expert_out.broadcast_mul(&selected_w)?;

                final_hidden = scatter_add(&final_hidden, &indices, &weighted)?;
            }
        }

        final_hidden.reshape((b, s, h))
    }
}

/// Return 1-D tensor of indices where mask is true.
fn compact_indices(mask: &Tensor) -> CandleResult<Tensor> {
    // Host-side compaction for Phase-1 correctness on both CPU and CUDA.
    // Production path will use a pure-device nonzero / exclusive-scan kernel.
    let mask_vec = mask.to_dtype(DType::U8)?.flatten_all()?.to_vec1::<u8>()?;
    let mut idxs = Vec::with_capacity(mask_vec.len());
    for (i, &m) in mask_vec.iter().enumerate() {
        if m != 0 {
            idxs.push(i as u32);
        }
    }
    let n = idxs.len();
    Tensor::from_vec(idxs, (n,), mask.device())
}

/// final[indices[i], :] += values[i, :]
fn scatter_add(base: &Tensor, indices: &Tensor, values: &Tensor) -> CandleResult<Tensor> {
    let h = values.dim(1)?;
    let n = indices.dims1()?;
    let indices_exp = indices.unsqueeze(1)?.expand((n, h))?.contiguous()?;
    base.scatter_add(&indices_exp, values, 0)
}
