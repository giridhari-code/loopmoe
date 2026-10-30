use candle_core::{Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use crate::config::ModelConfig;
use crate::cache::KvCache;
use super::{Attention, MoeLayer, rope::RotaryEmbedding};

pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    pub fn load(size: usize, eps: f32, vb: VarBuilder) -> CandleResult<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps: eps as f64 })
    }

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let x_dtype = x.dtype();
        let x = x.to_dtype(candle_core::DType::F32)?;
        let norm_x = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

pub struct TransformerBlock {
    pub self_attn: Attention,
    pub mlp: MoeLayer,
    pub input_layernorm: RmsNorm,
    pub post_attention_layernorm: RmsNorm,
}

impl TransformerBlock {
    pub fn load(
        vb: VarBuilder,
        config: &ModelConfig,
        rope: RotaryEmbedding,
    ) -> CandleResult<Self> {
        Ok(Self {
            self_attn: Attention::load(vb.pp("self_attn"), config, rope)?,
            mlp: MoeLayer::load(vb.pp("mlp"), config)?,
            input_layernorm: RmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: RmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut KvCache>,
        update_cache: bool,
    ) -> CandleResult<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self.self_attn.forward(&x, start_pos, cache, update_cache)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}
