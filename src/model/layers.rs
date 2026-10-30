use candle_core::{Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use crate::config::ModelConfig;
use super::{Attention, MoeLayer};

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(size: usize, eps: f32, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self {
            weight,
            eps: eps as f64,
        })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // x: [batch, seq, hidden]
        let x_dtype = x.dtype();
        let x = x.to_dtype(candle_core::DType::F32)?;
        let norm_x = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed
            .to_dtype(x_dtype)?
            .broadcast_mul(&self.weight)
    }
}

pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MoeLayer,          // MoE instead of dense FFN
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl TransformerBlock {
    pub fn load(vb: VarBuilder, config: &ModelConfig) -> CandleResult<Self> {
        let self_attn = Attention::load(vb.pp("self_attn"), config)?;
        let mlp = MoeLayer::load(vb.pp("mlp"), config)?;
        let input_layernorm = RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = RmsNorm::load(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> CandleResult<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, start_pos)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}
