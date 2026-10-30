use candle_core::{Result as CandleResult, Tensor};
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

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

/// MoE layer – Phase 1 placeholder
/// Real implementation will use efficient grouped GEMM + DeepEP style dispatch
/// inspired by DeepInfer kernels.
pub struct MoeLayer {
    gate: Linear,
    experts: Vec<Expert>,
    num_experts_per_tok: usize,
    num_experts: usize,
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
        })
    }

    /// Placeholder forward – uses first expert only for scaffold correctness.
    /// TODO: implement proper top-k routing + expert parallel dispatch.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // For scaffold we just run expert 0 so the model can compile and run.
        // Real top-k + weighted sum will be added next.
        self.experts[0].forward(x)
    }
}
