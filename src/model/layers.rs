use candle_core::{Result as CandleResult, Tensor};
use candle_nn::VarBuilder;
use crate::config::ModelConfig;
use crate::cache::{KvCache, MlaCache, RuntimeCaches};
use crate::kernels::cuda::{residual_add, rmsnorm as fused_rmsnorm};
use crate::kernels::force_diff_ops;
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

    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Training: pure Candle path (CUDA fused ops are no_bwd).
        if force_diff_ops() {
            return candle_nn::ops::rms_norm(x, &self.weight, self.eps);
        }
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

    fn add_residual(x: &Tensor, residual: &Tensor) -> CandleResult<Tensor> {
        if force_diff_ops() {
            Ok((x + residual)?)
        } else {
            residual_add(x, residual)
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        gqa: Option<&mut KvCache>,
        mla: Option<&mut MlaCache>,
        update_cache: bool,
    ) -> CandleResult<Tensor> {
        let residual = x;
        let x = self.input_layernorm.forward(x)?;
        let x = self
            .self_attn
            .forward(&x, start_pos, gqa, mla, update_cache)?;
        let x = Self::add_residual(&x, residual)?;
        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Self::add_residual(&x, residual)
    }

    /// Decode step for B sequences: batched norm/MLP, per-row attention+cache.
    pub fn forward_decode_batch(
        &self,
        x: &Tensor,
        positions: &[usize],
        caches: &mut [&mut RuntimeCaches],
        layer_idx: usize,
        update_cache: bool,
    ) -> CandleResult<Tensor> {
        let (b, _s, _h) = x.dims3()?;
        assert_eq!(b, positions.len());
        assert_eq!(b, caches.len());

        let residual = x;
        let x_n = self.input_layernorm.forward(x)?;

        let mut attn_rows = Vec::with_capacity(b);
        for i in 0..b {
            let row = x_n.narrow(0, i, 1)?;
            let out = match caches[i] {
                RuntimeCaches::Gqa(bank) => self.self_attn.forward(
                    &row,
                    positions[i],
                    Some(&mut bank.layers[layer_idx]),
                    None,
                    update_cache,
                )?,
                RuntimeCaches::Mla(bank) => self.self_attn.forward(
                    &row,
                    positions[i],
                    None,
                    Some(&mut bank.layers[layer_idx]),
                    update_cache,
                )?,
            };
            attn_rows.push(out);
        }
        let x = Tensor::cat(&attn_rows, 0)?;
        let x = Self::add_residual(&x, residual)?;

        let residual = &x;
        let x = self.post_attention_layernorm.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        Self::add_residual(&x, residual)
    }
}
