//! Trainable Sentinel v1.0 (candle VarMap).
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Result as CResult, Tensor, D};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder, VarMap};
use crate::config::ModelConfig;
use crate::model::rope::RotaryEmbedding;

struct Expert { w1: Linear, w2: Linear, w3: Linear }
impl Expert {
    fn load(vb: VarBuilder, hidden: usize, inter: usize) -> CResult<Self> {
        Ok(Self {
            w1: linear_no_bias(hidden, inter, vb.pp("w1"))?,
            w2: linear_no_bias(inter, hidden, vb.pp("w2"))?,
            w3: linear_no_bias(hidden, inter, vb.pp("w3"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let g = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        self.w2.forward(&(g * self.w3.forward(x)?)?)
    }
}

struct Moe {
    gate: Linear, routed: Vec<Expert>, shared: Vec<Expert>,
    top_k: usize, scale: f64, last_aux: f64,
}
impl Moe {
    fn load(vb: VarBuilder, cfg: &ModelConfig) -> CResult<Self> {
        let gate = linear_no_bias(cfg.hidden_size, cfg.n_routed_experts, vb.pp("gate"))?;
        let mut routed = Vec::new();
        for i in 0..cfg.n_routed_experts {
            routed.push(Expert::load(vb.pp(&format!("experts.{i}")), cfg.hidden_size, cfg.moe_intermediate_size)?);
        }
        let mut shared = Vec::new();
        for i in 0..cfg.n_shared_experts {
            shared.push(Expert::load(vb.pp(&format!("shared_experts.{i}")), cfg.hidden_size, cfg.moe_intermediate_size)?);
        }
        Ok(Self { gate, routed, shared, top_k: cfg.num_experts_per_tok, scale: cfg.routed_scaling_factor as f64, last_aux: 0.0 })
    }
    fn forward(&mut self, x: &Tensor) -> CResult<Tensor> {
        let (b, s, h) = x.dims3()?;
        let flat = x.reshape((b * s, h))?;
        let logits = self.gate.forward(&flat)?;
        let (topk_w, topk_ids) = crate::kernels::fertile::topk_last(&logits, self.top_k)?;
        let topk_w = candle_nn::ops::softmax_last_dim(&topk_w)?;
        let mut out = Tensor::zeros((b * s, h), x.dtype(), x.device())?;
        for e in &self.shared { out = (out + e.forward(&flat)?)?; }
        let t = b * s;
        let ids = topk_ids.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let ws = topk_w.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        let n_e = self.routed.len();
        let mut buckets: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n_e];
        for tok in 0..t {
            for k in 0..self.top_k {
                let e = ids[tok][k] as usize;
                if e < n_e { buckets[e].push((tok, ws[tok][k] * self.scale as f32)); }
            }
        }
        let hits: Vec<f64> = buckets.iter().map(|b| b.len() as f64).collect();
        let total: f64 = hits.iter().sum::<f64>().max(1.0);
        self.last_aux = hits.iter().map(|h| (h / total).powi(2) * n_e as f64).sum();
        let mut routed_acc = Tensor::zeros((t, h), x.dtype(), x.device())?;
        for (e, pairs) in buckets.into_iter().enumerate() {
            if pairs.is_empty() { continue; }
            let n = pairs.len();
            let mut toks = Vec::with_capacity(n);
            let mut wts = Vec::with_capacity(n);
            for (ti, w) in pairs { toks.push(ti as u32); wts.push(w); }
            let idx = Tensor::from_vec(toks, (n,), x.device())?;
            let w = Tensor::from_vec(wts, (n, 1), x.device())?.to_dtype(x.dtype())?;
            let y = self.routed[e].forward(&flat.index_select(&idx, 0)?)?.broadcast_mul(&w)?;
            let hdim = y.dim(1)?;
            let idx_exp = idx.unsqueeze(1)?.expand((n, hdim))?.contiguous()?;
            routed_acc = routed_acc.scatter_add(&idx_exp, &y, 0)?;
        }
        (out + routed_acc)?.reshape((b, s, h))
    }
}

struct Attn { q: Linear, k: Linear, v: Linear, o: Linear, n_heads: usize, n_kv: usize, head_dim: usize, rope: RotaryEmbedding }
impl Attn {
    fn load(vb: VarBuilder, cfg: &ModelConfig) -> CResult<Self> {
        let hd = cfg.head_dim;
        Ok(Self {
            q: linear_no_bias(cfg.hidden_size, cfg.num_attention_heads * hd, vb.pp("q_proj"))?,
            k: linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * hd, vb.pp("k_proj"))?,
            v: linear_no_bias(cfg.hidden_size, cfg.num_key_value_heads * hd, vb.pp("v_proj"))?,
            o: linear_no_bias(cfg.num_attention_heads * hd, cfg.hidden_size, vb.pp("o_proj"))?,
            n_heads: cfg.num_attention_heads, n_kv: cfg.num_key_value_heads, head_dim: hd,
            rope: RotaryEmbedding::new(hd, cfg.max_seq_len.min(cfg.max_position_embeddings), cfg.rope_theta, vb.device(), vb.dtype())?,
        })
    }
    fn forward(&self, x: &Tensor) -> CResult<Tensor> {
        let (b, s, _) = x.dims3()?;
        let q = self.q.forward(x)?.reshape((b, s, self.n_heads, self.head_dim))?.transpose(1, 2)?;
        let q = self.rope.forward(&q, 0)?;
        let mut k = self.k.forward(x)?.reshape((b, s, self.n_kv, self.head_dim))?.transpose(1, 2)?;
        k = self.rope.forward(&k, 0)?;
        let mut v = self.v.forward(x)?.reshape((b, s, self.n_kv, self.head_dim))?.transpose(1, 2)?;
        if self.n_kv != self.n_heads {
            let rep = self.n_heads / self.n_kv;
            k = k.unsqueeze(2)?.expand((b, self.n_kv, rep, s, self.head_dim))?.reshape((b, self.n_heads, s, self.head_dim))?;
            v = v.unsqueeze(2)?.expand((b, self.n_kv, rep, s, self.head_dim))?.reshape((b, self.n_heads, s, self.head_dim))?;
        }
        let scale = 1f64 / (self.head_dim as f64).sqrt();
        let att = (q.matmul(&k.transpose(D::Minus1, D::Minus2)?)? * scale)?;
        let mut m = vec![0f32; s * s];
        for i in 0..s { for j in (i + 1)..s { m[i * s + j] = f32::NEG_INFINITY; } }
        let mask = Tensor::from_vec(m, (1, 1, s, s), x.device())?;
        let att = candle_nn::ops::softmax_last_dim(&att.broadcast_add(&mask)?)?;
        let out = att.matmul(&v)?.transpose(1, 2)?.reshape((b, s, self.n_heads * self.head_dim))?;
        self.o.forward(&out)
    }
}

struct Block { ln1: RmsNorm, attn: Attn, ln2: RmsNorm, moe: Moe }
impl Block {
    fn load(vb: VarBuilder, cfg: &ModelConfig) -> CResult<Self> {
        Ok(Self {
            ln1: rms_norm(cfg.hidden_size, cfg.rms_norm_eps as f64, vb.pp("input_layernorm"))?,
            attn: Attn::load(vb.pp("self_attn"), cfg)?,
            ln2: rms_norm(cfg.hidden_size, cfg.rms_norm_eps as f64, vb.pp("post_attention_layernorm"))?,
            moe: Moe::load(vb.pp("mlp"), cfg)?,
        })
    }
    fn forward(&mut self, x: &Tensor) -> CResult<Tensor> {
        let h = (x + self.attn.forward(&self.ln1.forward(x)?)?)?;
        let m = self.moe.forward(&self.ln2.forward(&h)?)?;
        h + m
    }
}

pub struct SentinelTrainModel {
    pub varmap: VarMap,
    embed: Embedding,
    layers: Vec<Block>,
    norm: RmsNorm,
    lm_head: Linear,
    pub cfg: ModelConfig,
    pub aux_coef: f64,
}

impl SentinelTrainModel {
    pub fn new(cfg: &ModelConfig, device: &Device, dtype: DType) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        let embed = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("model.embed_tokens"))?;
        let mut layers = Vec::new();
        for i in 0..cfg.num_layers {
            layers.push(Block::load(vb.pp(&format!("model.layers.{i}")), cfg)?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps as f64, vb.pp("model.norm"))?;
        let lm_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;
        Ok(Self { varmap, embed, layers, norm, lm_head, cfg: cfg.clone(), aux_coef: 0.001 })
    }

    /// Forward pass returning logits and optional CE(+aux) loss.
    pub fn forward(&mut self, input_ids: &Tensor, labels: Option<&Tensor>) -> CResult<(Tensor, Option<Tensor>)> {
        let logits = self.forward_logits(input_ids)?;
        let loss = if let Some(labels) = labels {
            let logits_s = logits.i((.., 0..logits.dim(1)? - 1, ..))?;
            let labels_s = labels.i((.., 1..labels.dim(1)?))?;
            let (b, t, v) = logits_s.dims3()?;
            let ce = masked_ce(&logits_s.reshape((b * t, v))?, &labels_s.reshape((b * t,))?)?;
            let total = if self.aux_coef > 0.0 {
                // avg aux from last forward_logits call is not tracked here;
                // aux is accumulated inside forward_logits via layers — recompute cheaply from last_aux
                let mut aux = 0.0f64;
                let mut n = 0usize;
                for layer in &self.layers {
                    aux += layer.moe.last_aux;
                    n += 1;
                }
                // loop_count already applied inside forward_logits; last_aux is per last loop
                let avg = if n > 0 { aux / n as f64 } else { 0.0 };
                if avg > 0.0 {
                    let aux_t = Tensor::new(avg as f32, ce.device())?.to_dtype(ce.dtype())?;
                    (ce + (aux_t * self.aux_coef)?)?
                } else {
                    ce
                }
            } else {
                ce
            };
            Some(total)
        } else {
            None
        };
        Ok((logits, loss))
    }

    /// Shared backbone → logits [B, S, V].
    fn forward_logits(&mut self, input_ids: &Tensor) -> CResult<Tensor> {
        let mut x = self.embed.forward(input_ids)?;
        for _ in 0..self.cfg.loop_count {
            for layer in self.layers.iter_mut() {
                x = layer.forward(&x)?;
            }
        }
        self.lm_head.forward(&self.norm.forward(&x)?)
    }

    /// Sum of token log-probs for DPO: Σ log π(y_t | y_<t) over non-masked labels.
    /// Returns scalar tensor (batch mean of sequence log-likelihoods).
    pub fn sequence_logprob(&mut self, input_ids: &Tensor, labels: &Tensor) -> CResult<Tensor> {
        let logits = self.forward_logits(input_ids)?;
        let logits_s = logits.i((.., 0..logits.dim(1)? - 1, ..))?;
        let labels_s = labels.i((.., 1..labels.dim(1)?))?;
        let (b, t, v) = logits_s.dims3()?;
        sum_logprob(&logits_s.reshape((b * t, v))?, &labels_s.reshape((b * t,))?, b)
    }

    pub fn parameters(&self) -> Result<Vec<candle_core::Var>> {
        Ok(self.varmap.all_vars())
    }
}

/// Cross-entropy that ignores label -100 (mean NLL).
fn masked_ce(logits: &Tensor, labels: &Tensor) -> CResult<Tensor> {
    let (lp, mask, cnt) = logprob_parts(logits, labels)?;
    lp.neg()?.mul(&mask)?.sum_all()? / cnt
}

/// Mean sequence log-likelihood (sum of log p over tokens, averaged over batch).
fn sum_logprob(logits: &Tensor, labels: &Tensor, batch: usize) -> CResult<Tensor> {
    let (lp, mask, _cnt) = logprob_parts(logits, labels)?;
    // sum over all tokens, then mean over batch
    let total = lp.mul(&mask)?.sum_all()?;
    total / (batch as f64).max(1.0)
}

fn logprob_parts(logits: &Tensor, labels: &Tensor) -> CResult<(Tensor, Tensor, f64)> {
    let dev = logits.device();
    let lab = labels.to_vec1::<i64>()?;
    let safe: Vec<u32> = lab.iter().map(|&l| if l < 0 { 0 } else { l as u32 }).collect();
    let mask: Vec<f32> = lab.iter().map(|&l| if l < 0 { 0.0 } else { 1.0 }).collect();
    let n = lab.len();
    let cnt = mask.iter().sum::<f32>().max(1.0) as f64;
    let safe = Tensor::from_vec(safe, (n, 1), dev)?;
    let mask = Tensor::from_vec(mask, (n,), dev)?;
    let lp = candle_nn::ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)?;
    let picked = lp.gather(&safe, 1)?.squeeze(1)?;
    Ok((picked, mask, cnt))
}
