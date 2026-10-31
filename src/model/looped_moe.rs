use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Module, VarBuilder};
use crate::config::ModelConfig;
use crate::cache::RuntimeCaches;
use super::traits::InferModel;
use super::{RmsNorm, TransformerBlock, rope::RotaryEmbedding};

pub struct LoopedMoE {
    pub config: ModelConfig,
    pub embed_tokens: candle_nn::Embedding,
    pub layers: Vec<TransformerBlock>,
    pub norm: RmsNorm,
    pub lm_head: candle_nn::Linear,
    pub device: Device,
}

impl LoopedMoE {
    pub fn load(vb: VarBuilder, config: &ModelConfig, device: &Device) -> CandleResult<Self> {
        let dtype = vb.dtype();
        let rope_len = config.max_seq_len.min(config.max_position_embeddings);
        let rope_dim = if config.use_mla {
            config.qk_rope_head_dim
        } else {
            config.head_dim
        };
        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("model.embed_tokens"))?;
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer_rope =
                RotaryEmbedding::new(rope_dim, rope_len, config.rope_theta, device, dtype)?;
            layers.push(TransformerBlock::load(
                vb.pp(&format!("model.layers.{i}")),
                config,
                layer_rope,
            )?);
        }
        let norm = RmsNorm::load(config.hidden_size, config.rms_norm_eps, vb.pp("model.norm"))?;
        let lm_head = if config.tie_word_embeddings {
            match candle_nn::linear(config.hidden_size, config.vocab_size, vb.pp("lm_head")) {
                Ok(l) => l,
                Err(_) => candle_nn::linear(
                    config.hidden_size,
                    config.vocab_size,
                    vb.pp("model.embed_tokens"),
                )?,
            }
        } else {
            candle_nn::linear(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        tracing::info!(
            layers = config.num_layers,
            loop_count = config.loop_count,
            depth = config.num_layers * config.loop_count,
            mla = config.use_mla,
            experts = config.n_routed_experts,
            "LoopedMoE loaded"
        );
        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        mut caches: Option<&mut RuntimeCaches>,
    ) -> CandleResult<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        let num_loops = self.config.loop_count;
        for loop_idx in 0..num_loops {
            let update_cache = loop_idx == num_loops - 1;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                match caches.as_deref_mut() {
                    Some(RuntimeCaches::Gqa(bank)) => {
                        let cache = &mut bank.layers[layer_idx];
                        hidden =
                            layer.forward(&hidden, start_pos, Some(cache), None, update_cache)?;
                    }
                    Some(RuntimeCaches::Mla(bank)) => {
                        let cache = &mut bank.layers[layer_idx];
                        hidden =
                            layer.forward(&hidden, start_pos, None, Some(cache), update_cache)?;
                    }
                    None => {
                        hidden = layer.forward(&hidden, start_pos, None, None, false)?;
                    }
                }
            }
        }
        let hidden = self.norm.forward(&hidden)?;
        let hidden = if seq_len > 1 {
            hidden.narrow(1, seq_len - 1, 1)?.squeeze(1)?
        } else {
            hidden.squeeze(1)?
        };
        self.lm_head.forward(&hidden)
    }

    /// O3 helper: run a prompt slice `[start_pos, start_pos + tokens)` into cache.
    /// Returns last-position logits (sample only when this is the final chunk).
    pub fn forward_prefill_chunk(
        &self,
        tokens: &[u32],
        start_pos: usize,
        caches: &mut RuntimeCaches,
    ) -> CandleResult<Tensor> {
        if tokens.is_empty() {
            candle_core::bail!("forward_prefill_chunk: empty tokens");
        }
        let input = Tensor::new(tokens, &self.device)?.reshape((1, tokens.len()))?;
        self.forward(&input, start_pos, Some(caches))
    }

    /// O2.1: one decode step for `B` sequences → logits `[B, vocab]`.
    pub fn forward_decode_batch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        caches: &mut [&mut RuntimeCaches],
    ) -> CandleResult<Tensor> {
        let b = tokens.len();
        if b == 0 {
            candle_core::bail!("forward_decode_batch: empty");
        }
        if positions.len() != b || caches.len() != b {
            candle_core::bail!(
                "forward_decode_batch: len mismatch tokens={} pos={} caches={}",
                b,
                positions.len(),
                caches.len()
            );
        }

        if b == 1 {
            let input = Tensor::new(tokens, &self.device)?.reshape((1, 1))?;
            let logits = self.forward(&input, positions[0], Some(caches[0]))?;
            return match logits.dims() {
                [_] => logits.unsqueeze(0),
                [1, _] => Ok(logits),
                _ => logits.reshape((1, logits.elem_count())),
            };
        }

        let input = Tensor::new(tokens, &self.device)?.reshape((b, 1))?;
        let mut hidden = self.embed_tokens.forward(&input)?;

        let num_loops = self.config.loop_count;
        for loop_idx in 0..num_loops {
            let update_cache = loop_idx == num_loops - 1;
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                hidden = layer.forward_decode_batch(
                    &hidden,
                    positions,
                    caches,
                    layer_idx,
                    update_cache,
                )?;
            }
        }

        let hidden = self.norm.forward(&hidden)?;
        let hidden = hidden.squeeze(1)?;
        self.lm_head.forward(&hidden)
    }
}

impl InferModel for LoopedMoE {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        start_pos: usize,
        caches: Option<&mut RuntimeCaches>,
    ) -> CandleResult<Tensor> {
        LoopedMoE::forward(self, input_ids, start_pos, caches)
    }

    fn forward_decode_batch(
        &self,
        tokens: &[u32],
        positions: &[usize],
        caches: &mut [&mut RuntimeCaches],
    ) -> CandleResult<Tensor> {
        LoopedMoE::forward_decode_batch(self, tokens, positions, caches)
    }
}
