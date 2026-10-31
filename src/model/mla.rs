//! Multi-head Latent Attention (DeepSeek MLA) — DoBu-fused scores path.
use candle_core::{Result as CandleResult, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use crate::config::ModelConfig;
use crate::cache::MlaCache;
use crate::kernels::dobu::{dobu_mla_prefill, DobuMlaParams};
use super::layers::RmsNorm;
use super::rope::RotaryEmbedding;

pub struct MlaAttention {
    q_a_proj: Option<Linear>,
    q_a_layernorm: Option<RmsNorm>,
    q_b_proj: Linear,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    rope: RotaryEmbedding,
    softmax_scale: f64,
}

impl MlaAttention {
    pub fn load(vb: VarBuilder, config: &ModelConfig, rope: RotaryEmbedding) -> CandleResult<Self> {
        let n = config.num_attention_heads;
        let qk_nope = config.qk_nope_head_dim;
        let qk_rope = config.qk_rope_head_dim;
        let v_dim = config.v_head_dim;
        let kv_rank = config.kv_lora_rank;
        let qk_head = qk_nope + qk_rope;

        let (q_a_proj, q_a_layernorm, q_b_proj) = if config.q_lora_rank > 0 {
            (
                Some(linear_no_bias(config.hidden_size, config.q_lora_rank, vb.pp("q_a_proj"))?),
                Some(RmsNorm::load(config.q_lora_rank, config.rms_norm_eps, vb.pp("q_a_layernorm"))?),
                linear_no_bias(config.q_lora_rank, n * qk_head, vb.pp("q_b_proj"))?,
            )
        } else {
            (None, None, linear_no_bias(config.hidden_size, n * qk_head, vb.pp("q_proj"))?)
        };

        let kv_a = linear_no_bias(config.hidden_size, kv_rank + qk_rope, vb.pp("kv_a_proj_with_mqa"))?;
        let kv_a_ln = RmsNorm::load(kv_rank, config.rms_norm_eps, vb.pp("kv_a_layernorm"))?;
        let kv_b = linear_no_bias(kv_rank, n * (qk_nope + v_dim), vb.pp("kv_b_proj"))?;
        let o_proj = linear_no_bias(n * v_dim, config.hidden_size, vb.pp("o_proj"))?;
        let scale = 1.0 / ((qk_nope + qk_rope) as f64).sqrt();

        Ok(Self {
            q_a_proj,
            q_a_layernorm,
            q_b_proj,
            kv_a_proj_with_mqa: kv_a,
            kv_a_layernorm: kv_a_ln,
            kv_b_proj: kv_b,
            o_proj,
            num_heads: n,
            qk_nope_head_dim: qk_nope,
            qk_rope_head_dim: qk_rope,
            v_head_dim: v_dim,
            kv_lora_rank: kv_rank,
            rope,
            softmax_scale: scale,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut MlaCache>,
        update_cache: bool,
    ) -> CandleResult<Tensor> {
        let (b, s, _) = x.dims3()?;
        let n = self.num_heads;
        let nope = self.qk_nope_head_dim;
        let rope_d = self.qk_rope_head_dim;
        let v_d = self.v_head_dim;
        let rank = self.kv_lora_rank;

        let q = if let (Some(qa), Some(ln)) = (&self.q_a_proj, &self.q_a_layernorm) {
            let c = qa.forward(x)?;
            self.q_b_proj.forward(&ln.forward(&c)?)?
        } else {
            self.q_b_proj.forward(x)?
        };
        let q = q.reshape((b, s, n, nope + rope_d))?;
        let q_nope = q.narrow(D::Minus1, 0, nope)?.transpose(1, 2)?;
        let q_pe = self
            .rope
            .forward(&q.narrow(D::Minus1, nope, rope_d)?.transpose(1, 2)?, start_pos)?;

        let kv = self.kv_a_proj_with_mqa.forward(x)?;
        let c_kv = self.kv_a_layernorm.forward(&kv.narrow(D::Minus1, 0, rank)?)?;
        let k_pe = self
            .rope
            .forward(&kv.narrow(D::Minus1, rank, rope_d)?.unsqueeze(1)?, start_pos)?;

        let (c_full, k_pe_full) = if let Some(cache) = cache {
            if update_cache {
                cache.append(&c_kv, &k_pe)?;
                (
                    cache.c_view()?.expect("mla c"),
                    cache.k_pe_view()?.expect("mla k_pe"),
                )
            } else {
                match (cache.c_view()?, cache.k_pe_view()?) {
                    (Some(pc), Some(pk)) => {
                        (Tensor::cat(&[&pc, &c_kv], 1)?, Tensor::cat(&[&pk, &k_pe], 2)?)
                    }
                    _ => (c_kv, k_pe),
                }
            }
        } else {
            (c_kv, k_pe)
        };

        let s_full = c_full.dim(1)?;
        let kv_up = self
            .kv_b_proj
            .forward(&c_full)?
            .reshape((b, s_full, n, nope + v_d))?;
        let k_nope = kv_up.narrow(D::Minus1, 0, nope)?.transpose(1, 2)?;
        let v = kv_up.narrow(D::Minus1, nope, v_d)?.transpose(1, 2)?;
        let k_pe_h = k_pe_full.broadcast_as((b, n, s_full, rope_d))?;

        let q = Tensor::cat(&[&q_nope, &q_pe], D::Minus1)?;
        let k = Tensor::cat(&[&k_nope, &k_pe_h], D::Minus1)?;

        // DoBu MLA path (explicit scale — no ambiguous default)
        let params = DobuMlaParams {
            causal: s > 1 && start_pos == 0,
            softmax_scale: Some(self.softmax_scale),
        };
        let out = dobu_mla_prefill(&q, &k, &v, &params)?;
        let out = out.transpose(1, 2)?.reshape((b, s, n * v_d))?;
        self.o_proj.forward(&out)
    }
}
