use candle_core::{Result as CandleResult, Tensor, D};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use crate::config::ModelConfig;
use crate::cache::KvCache;
use super::rope::RotaryEmbedding;

pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: RotaryEmbedding,
}

impl Attention {
    pub fn load(
        vb: VarBuilder,
        config: &ModelConfig,
        rope: RotaryEmbedding,
    ) -> CandleResult<Self> {
        let head_dim = config.head_dim;
        Ok(Self {
            q_proj: linear_no_bias(
                config.hidden_size,
                config.num_attention_heads * head_dim,
                vb.pp("q_proj"),
            )?,
            k_proj: linear_no_bias(
                config.hidden_size,
                config.num_key_value_heads * head_dim,
                vb.pp("k_proj"),
            )?,
            v_proj: linear_no_bias(
                config.hidden_size,
                config.num_key_value_heads * head_dim,
                vb.pp("v_proj"),
            )?,
            o_proj: linear_no_bias(
                config.num_attention_heads * head_dim,
                config.hidden_size,
                vb.pp("o_proj"),
            )?,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            rope,
        })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        cache: Option<&mut KvCache>,
        update_cache: bool,
    ) -> CandleResult<Tensor> {
        let (b, s, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, s, self.num_heads, self.head_dim))?.transpose(1, 2)?;
        let k = k.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;
        let v = v.reshape((b, s, self.num_kv_heads, self.head_dim))?.transpose(1, 2)?;

        let q = self.rope.forward(&q, start_pos)?;
        let k = self.rope.forward(&k, start_pos)?;

        let (k_full, v_full) = if let Some(cache) = cache {
            let k_full = match &cache.k {
                Some(prev) => Tensor::cat(&[prev, &k], 2)?,
                None => k.clone(),
            };
            let v_full = match &cache.v {
                Some(prev) => Tensor::cat(&[prev, &v], 2)?,
                None => v.clone(),
            };
            if update_cache {
                cache.append(&k, &v)?;
            }
            (k_full, v_full)
        } else {
            (k, v)
        };

        let k_full = repeat_kv(k_full, self.num_heads / self.num_kv_heads)?;
        let v_full = repeat_kv(v_full, self.num_heads / self.num_kv_heads)?;

        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k_full.transpose(D::Minus1, D::Minus2)?)? * scale)?;

        let att = if s > 1 {
            let kv_len = k_full.dim(2)?;
            let mask = causal_mask(s, kv_len, q.device())?;
            let att = att.broadcast_add(&mask)?;
            candle_nn::ops::softmax_last_dim(&att)?
        } else {
            candle_nn::ops::softmax_last_dim(&att)?
        };

        let out = att.matmul(&v_full)?;
        let out = out.transpose(1, 2)?.reshape((b, s, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}

fn repeat_kv(x: Tensor, n_rep: usize) -> CandleResult<Tensor> {
    if n_rep == 1 {
        return Ok(x);
    }
    let (b, h, s, d) = x.dims4()?;
    x.unsqueeze(2)?.expand((b, h, n_rep, s, d))?.reshape((b, h * n_rep, s, d))
}

fn causal_mask(q_len: usize, kv_len: usize, device: &candle_core::Device) -> CandleResult<Tensor> {
    let mut mask = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        let max_key = kv_len - q_len + i;
        for j in 0..kv_len {
            if j > max_key {
                mask[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    Tensor::from_vec(mask, (1, 1, q_len, kv_len), device)
}
