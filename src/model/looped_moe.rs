use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Module, VarBuilder};
use crate::config::ModelConfig;
use super::{Attention, MoeLayer, RmsNorm, TransformerBlock};

/// Looped MoE model.
///
/// Architecture:
/// - Embedding
/// - Shared TransformerBlock (contains Attention + MoE) applied `loop_count` times
/// - Final RMSNorm + LM head
///
/// This is the core of weight-tied recurrent depth + sparse MoE.
pub struct LoopedMoE {
    pub config: ModelConfig,
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<TransformerBlock>,   // physical layers (shared across loops)
    pub norm: RmsNorm,
    pub lm_head: candle_nn::Linear,
    pub device: Device,
}

impl LoopedMoE {
    pub fn load(vb: VarBuilder, config: &ModelConfig, device: &Device) -> CandleResult<Self> {
        let embed_tokens = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            vb.pp("model.embed_tokens"),
        )?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = TransformerBlock::load(
                vb.pp(&format!("model.layers.{i}")),
                config,
            )?;
            layers.push(layer);
        }

        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        
        let lm_head = if config.tie_word_embeddings {
            // weight tying – will use embed_tokens weights later
            candle_nn::linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        } else {
            candle_nn::linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
        })
    }

    /// Forward pass with looping.
    /// For each of the `loop_count` iterations we run all physical layers.
    pub fn forward(&self, input_ids: &Tensor, start_pos: usize) -> CandleResult<Tensor> {
        let (_b_sz, seq_len) = input_ids.dims2()?;
        let mut hidden = self.embed_tokens.forward(input_ids)?;

        // === Looped application of the shared stack ===
        for _loop_idx in 0..self.config.loop_count {
            for layer in &self.layers {
                hidden = layer.forward(&hidden, start_pos)?;
            }
        }

        let hidden = self.norm.forward(&hidden)?;
        
        // Only last token for generation (common optimization)
        let hidden = if seq_len > 1 {
            hidden.i((.., seq_len - 1, ..))?
        } else {
            hidden
        };

        let logits = self.lm_head.forward(&hidden)?;
        Ok(logits)
    }
}
