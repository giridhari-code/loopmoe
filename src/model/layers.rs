use candle_core::{Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use crate::config::ModelConfig;
use crate::cache::{KvCache, MlaCache};
use crate::kernels::cuda::{residual_add, rmsnorm as fused_rmsnorm};
use super::{Attention, MoeLayer, MlaAttention, rope::RotaryEmbedding};

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

    /// Prefer fused CUDA RMSNorm (single kernel: reduce + rsqrt + scale).
    /// Falls back to Candle when PTX / CUDA unavailable.
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        fused_rmsnorm(x, &self.weight, self.eps)
    }
}

pub enum SelfAttention {
    Gqa(Attention),
    Mla(MlaAttention),
}

impl SelfAttention {
    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        gqa: Option<&mut KvCache>,
        mla: Option<&mut MlaCache>,
        update_cache: bool,
    ) -> CandleResult<Tensor> {
        match self {
            SelfAttention::Gqa(a) => a.forward(x, start_pos, gqa, update_cache),
            SelfAttention::Mla(a) => a.forward(x, start_pos, mla, update_cache),
        }
    }
}

pub struct TransformerBlock {
    input_layernorm: RmsNorm,
    self_attn: SelfAttention,
    post_attention_layernorm: RmsNorm,
    mlp: MoeLayer,
}

impl TransformerBlock {
    pub fn load(vb: VarBuilder, config: &ModelConfig, rope: RotaryEmbedding) -> CandleResult<Self> {
        let self_attn = if config.use_mla {
            SelfAttention::Mla(MlaAttention::load(vb.pp("self_attn"), config, rope)?)
        } else {
            SelfAttention::Gqa(Attention::load(vb.pp("self_attn"), config, rope)?)
        };
        Ok(Self {
            input_layernorm: RmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn,
            post_attention_layernorm: RmsNorm::load(
                config.hidden_size,
                config.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: MoeLayer::load(vb.pp("mlp"), config)?,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        gqa: Option<&mut KvCache>,
        mla: Option<&mut MlaCache>,
        update_cache: bool,
    ) -> CandleResult<Tensor> {
        // residual = x
        // x = rmsnorm(x)          ← fused CUDA
        // x = attn(x)
        // x = x + residual        ← fused residual_add
        // residual = x
        // x = rmsnorm(x)          ← fused CUDA
        // x = mlp(x)              ← silu_mul fused inside experts
        // x = x + residual        ← fused residual_add
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .self_attn
            .forward(&x, start_pos, gqa, mla, update_cache)?;
        let x = residual_add(&x, residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual_add(&x, residual)
    }
}
