use candle_core::{Result as CandleResult, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use crate::config::ModelConfig;
use crate::kernels::fertile::{
    fertile_moe, FertileExpert, FertileMoeParams, KERNEL_NAME,
};

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
}

impl FertileExpert for Expert {
    #[inline]
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

/// Thin wrapper — compute is Fertile.
pub struct MoeLayer {
    gate: Linear,
    routed_experts: Vec<Expert>,
    shared_experts: Vec<Expert>,
    params: FertileMoeParams,
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

        let params = FertileMoeParams {
            num_experts_per_tok: config.num_experts_per_tok,
            routed_scaling_factor: config.routed_scaling_factor,
            norm_topk_prob: config.norm_topk_prob,
            use_sigmoid: config.scoring_func == "sigmoid",
        };

        tracing::debug!(
            kernel = KERNEL_NAME,
            routed = config.n_routed_experts,
            shared = config.n_shared_experts,
            "MoE layer loaded"
        );

        Ok(Self {
            gate,
            routed_experts,
            shared_experts,
            params,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;
        let y = fertile_moe(
            &x_flat,
            &self.gate,
            &self.routed_experts,
            &self.shared_experts,
            self.params,
        )?;
        y.reshape((b, s, h))
    }
}
