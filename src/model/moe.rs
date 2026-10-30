use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use crate::config::ModelConfig;

/// Single expert (SwiGLU: silu(w1(x)) * w3(x) → w2)
pub struct Expert {
    w1: Linear,
    w2: Linear,
    w3: Linear,
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

/// DeepSeek / DeepInfer style MoE:
///   output = shared_experts(x) + routed_scaling_factor * sum (w_k * routed_expert_k(x))
pub struct MoeLayer {
    gate: Linear,
    routed_experts: Vec<Expert>,
    shared_experts: Vec<Expert>,
    num_experts_per_tok: usize,
    routed_scaling_factor: f32,
    scoring_func: String,
    norm_topk_prob: bool,
}

impl MoeLayer {
    pub fn load(vb: VarBuilder, config: &ModelConfig) -> CandleResult<Self> {
        let gate = linear_no_bias(
            config.hidden_size,
            config.n_routed_experts,
            vb.pp("gate"),
        )?;

        // experts.0 .. n_routed-1 = routed; experts.n_routed .. = shared
        let mut routed_experts = Vec::with_capacity(config.n_routed_experts);
        for i in 0..config.n_routed_experts {
            routed_experts.push(Expert::load(
                vb.pp(&format!("experts.{i}")),
                config.hidden_size,
                config.moe_intermediate_size,
            )?);
        }

        let mut shared_experts = Vec::with_capacity(config.n_shared_experts);
        for i in 0..config.n_shared_experts {
            let idx = config.n_routed_experts + i;
            shared_experts.push(Expert::load(
                vb.pp(&format!("experts.{idx}")),
                config.hidden_size,
                config.moe_intermediate_size,
            )?);
        }

        Ok(Self {
            gate,
            routed_experts,
            shared_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            routed_scaling_factor: config.routed_scaling_factor,
            scoring_func: config.scoring_func.clone(),
            norm_topk_prob: config.norm_topk_prob,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, s, h) = x.dims3()?;
        let num_tokens = b * s;
        let x_flat = x.reshape((num_tokens, h))?;

        // Shared experts (always on)
        let mut shared_out = Tensor::zeros((num_tokens, h), x.dtype(), x.device())?;
        for expert in &self.shared_experts {
            shared_out = (shared_out + expert.forward(&x_flat)?)?;
        }

        // Routed top-k
        let router_logits = self.gate.forward(&x_flat)?;
        let scores = match self.scoring_func.as_str() {
            "sigmoid" => candle_nn::ops::sigmoid(&router_logits)?,
            _ => router_logits.clone(),
        };

        let (topk_weight, topk_ids) = scores.topk(self.num_experts_per_tok)?;
        let topk_ids = topk_ids.to_dtype(DType::I64)?;

        let topk_weight = if self.scoring_func == "sigmoid" {
            if self.norm_topk_prob {
                let sum = topk_weight.sum_keepdim(1)?;
                topk_weight.broadcast_div(&(sum + 1e-20)?)?
            } else {
                topk_weight
            }
        } else {
            candle_nn::ops::softmax_last_dim(&topk_weight)?
        };

        let mut routed_out = Tensor::zeros((num_tokens, h), x.dtype(), x.device())?;

        for k in 0..self.num_experts_per_tok {
            let expert_ids_k = topk_ids.i((.., k))?;
            let weights_k = topk_weight.i((.., k))?;

            for expert_idx in 0..self.routed_experts.len() {
                let expert_id_val = Tensor::new(expert_idx as i64, expert_ids_k.device())?
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
                let expert_out = self.routed_experts[expert_idx].forward(&selected_x)?;
                let weighted = expert_out.broadcast_mul(&selected_w)?;
                routed_out = scatter_add(&routed_out, &indices, &weighted)?;
            }
        }

        let scale = Tensor::new(self.routed_scaling_factor, x.device())?.to_dtype(x.dtype())?;
        let routed_out = routed_out.broadcast_mul(&scale)?;
        let out = (shared_out + routed_out)?;
        out.reshape((b, s, h))
    }
}

fn compact_indices(mask: &Tensor) -> CandleResult<Tensor> {
    let mask_vec = mask.to_dtype(DType::U8)?.flatten_all()?.to_vec1::<u8>()?;
    let mut idxs = Vec::with_capacity(mask_vec.len());
    for (i, &m) in mask_vec.iter().enumerate() {
        if m != 0 {
            idxs.push(i as u32);
        }
    }
    Tensor::from_vec(idxs.clone(), (idxs.len(),), mask.device())
}

fn scatter_add(base: &Tensor, indices: &Tensor, values: &Tensor) -> CandleResult<Tensor> {
    let h = values.dim(1)?;
    let n = indices.dims1()?;
    let indices_exp = indices.unsqueeze(1)?.expand((n, h))?.contiguous()?;
    base.scatter_add(&indices_exp, values, 0)
}
