use candle_core::{DType, Result as CandleResult, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use crate::config::ModelConfig;

/// Single expert (SwiGLU)
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

    #[inline]
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

/// ============================================================
///  LoopMoE fused kernel path (ONE specialized path for this model)
///
///  Optimized for: Looped Transformer + Shared/Routed MoE
///
///  Formula (DeepSeek / DeepInfer style):
///    y = Shared(x) + scale * Σ_k w_k * RoutedExpert_k(x)
///
///  Optimizations vs naive version:
///    1. Shared experts run once (not inside loop)
///    2. All (token, expert, weight) pairs collected first
///    3. Each routed expert runs EXACTLY ONCE on its tokens
///    4. Single scatter-add pass per expert
///    5. No K×E nested expert launches
///
///  Later this whole path can be replaced by one CUDA kernel
///  (DeepEP-style dispatch + grouped GEMM).
/// ============================================================
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

    /// Fused MoE forward — the single optimized path for this model.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, s, h) = x.dims3()?;
        let t = b * s;
        let x_flat = x.reshape((t, h))?;

        // 1. Shared experts (always-on, one pass)
        let mut out = Tensor::zeros((t, h), x.dtype(), x.device())?;
        for expert in &self.shared_experts {
            out = (out + expert.forward(&x_flat)?)?;
        }

        // 2. Router + top-k
        let logits = self.gate.forward(&x_flat)?;
        let scores = match self.scoring_func.as_str() {
            "sigmoid" => candle_nn::ops::sigmoid(&logits)?,
            _ => logits,
        };

        let (mut weights, ids) = scores.topk(self.num_experts_per_tok)?;
        let ids = ids.to_dtype(DType::I64)?;

        weights = if self.scoring_func == "sigmoid" {
            if self.norm_topk_prob {
                let sum = weights.sum_keepdim(1)?;
                weights.broadcast_div(&(sum + 1e-20)?)?
            } else {
                weights
            }
        } else {
            candle_nn::ops::softmax_last_dim(&weights)?
        };

        // 3. Group tokens by expert (ONE pass)
        let ids_vec = ids.flatten_all()?.to_vec1::<i64>()?;
        let w_vec = weights.flatten_all()?.to_vec1::<f32>()?;
        let k = self.num_experts_per_tok;
        let n_routed = self.routed_experts.len();

        let mut buckets: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n_routed];
        for tok in 0..t {
            for slot in 0..k {
                let e = ids_vec[tok * k + slot] as usize;
                let w = w_vec[tok * k + slot];
                if e < n_routed {
                    buckets[e].push((tok as u32, w));
                }
            }
        }

        // 4. Each routed expert runs EXACTLY ONCE
        let mut routed = Tensor::zeros((t, h), x.dtype(), x.device())?;
        let scale = self.routed_scaling_factor;

        for (e, pairs) in buckets.into_iter().enumerate() {
            if pairs.is_empty() {
                continue;
            }
            let n = pairs.len();
            let mut tok_ids = Vec::with_capacity(n);
            let mut ws = Vec::with_capacity(n);
            for (ti, w) in pairs {
                tok_ids.push(ti);
                ws.push(w * scale);
            }

            let indices = Tensor::from_vec(tok_ids, (n,), x.device())?;
            let w_tensor = Tensor::from_vec(ws, (n, 1), x.device())?.to_dtype(x.dtype())?;

            let selected = x_flat.index_select(&indices, 0)?;
            let expert_out = self.routed_experts[e].forward(&selected)?;
            let weighted = expert_out.broadcast_mul(&w_tensor)?;

            routed = scatter_add(&routed, &indices, &weighted)?;
        }

        // 5. Combine
        let out = (out + routed)?;
        out.reshape((b, s, h))
    }
}

fn scatter_add(base: &Tensor, indices: &Tensor, values: &Tensor) -> CandleResult<Tensor> {
    let h = values.dim(1)?;
    let n = indices.dims1()?;
    let indices_exp = indices.unsqueeze(1)?.expand((n, h))?.contiguous()?;
    base.scatter_add(&indices_exp, values, 0)
}
