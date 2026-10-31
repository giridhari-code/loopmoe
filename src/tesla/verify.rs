//! Tesla verify — chunk path uses one sequence forward (heavy matrix mults once)
//! instead of N sequential single-token forwards.

use candle_core::{Device, IndexOp, Result as CResult, Tensor, D};
use crate::model::LoopedMoE;
use crate::cache::RuntimeCaches;
use crate::sampler::{sample_token, SamplingParams};
use rand::Rng;

#[derive(Debug, Default, Clone)]
pub struct TeslaStepStats {
    pub drafted: usize,
    pub scheduled: usize,
    pub accepted: usize,
    /// How many model.forward calls this step used (chunk = 1, sequential = n).
    pub forwards: usize,
}

impl TeslaStepStats {
    pub fn acceptance_rate(&self) -> f32 {
        if self.scheduled == 0 {
            0.0
        } else {
            self.accepted as f32 / self.scheduled as f32
        }
    }
}

fn squeeze_last(logits: &Tensor) -> CResult<Tensor> {
    match logits.dims() {
        [_] => Ok(logits.clone()),
        [1, _] => logits.squeeze(0),
        [1, 1, _] => logits.squeeze(0)?.squeeze(0),
        d if d.len() == 2 => logits.i(0),
        _ => logits.flatten_all(),
    }
}

/// Greedy argmax without full sampling stack (fast path for temperature ≈ 0).
fn argmax_token(logits: &Tensor) -> CResult<u32> {
    let flat = squeeze_last(logits)?;
    flat.argmax(D::Minus1)?.to_scalar::<u32>()
}

fn is_greedy(params: &SamplingParams) -> bool {
    params.temperature <= 0.0
        && params.top_k.is_none()
        && params.top_p.is_none()
        && (params.repetition_penalty - 1.0).abs() < 1e-6
}

/// Verify draft tokens against the target model.
///
/// **Chunk path (default):** restore KV → one forward on
/// `[prev_token, draft[0], …, draft[n-2]]` so attention/GEMM run as one big
/// matrix multiplication sequence → compare logits at each position.
/// On first mismatch emit target token and stop (standard speculative decoding).
pub fn verify_and_accept(
    model: &LoopedMoE,
    caches: &mut RuntimeCaches,
    lens_before_draft: &[usize],
    prev_token: u32,
    pos_before_draft: usize,
    draft: &[u32],
    verify_len: usize,
    device: &Device,
    params: &SamplingParams,
    generated: &[u32],
    rng: &mut impl Rng,
) -> CResult<(Vec<u32>, TeslaStepStats)> {
    caches.restore_lens(lens_before_draft);

    let n = verify_len.min(draft.len());
    let mut stats = TeslaStepStats {
        drafted: draft.len(),
        scheduled: n,
        accepted: 0,
        forwards: 0,
    };

    if n == 0 {
        let input = Tensor::new(&[prev_token], device)?.unsqueeze(0)?;
        let logits = squeeze_last(&model.forward(&input, pos_before_draft, Some(caches))?)?;
        stats.forwards = 1;
        let t = if is_greedy(params) {
            argmax_token(&logits)?
        } else {
            sample_token(&logits, params, generated, rng)?
        };
        stats.accepted = 1;
        return Ok((vec![t], stats));
    }

    // ---- Chunk verify: one sequence forward (matrix-heavy path) ----
    // Input tokens: prev + draft[0..n-1]  → logits at positions predict draft[0], draft[1], …
    // After accepting k draft tokens we still need one extra target sample at the end
    // when all match (bonus token), or the rejection token when mismatch.
    let mut seq: Vec<u32> = Vec::with_capacity(n);
    seq.push(prev_token);
    for i in 0..(n - 1) {
        seq.push(draft[i]);
    }
    // seq length = n  → n logit rows predicting draft[0]..draft[n-1] conceptually
    // actually: position i predicts next token after seq[i]

    let input = Tensor::new(seq.as_slice(), device)?.unsqueeze(0)?; // (1, n)
    let logits = model.forward(&input, pos_before_draft, Some(caches))?; // (1, n, V) or (1, V)
    stats.forwards = 1;

    // Normalize to (seq_len, vocab)
    let logits_2d = match logits.dims() {
        [1, s, v] if *s > 1 => logits.squeeze(0)?,           // (s, v)
        [1, v] => logits.squeeze(0)?.unsqueeze(0)?,          // (1, v)
        [s, v] if *s > 1 => logits.clone(),
        [_] => logits.unsqueeze(0)?,
        _ => logits.flatten(0, logits.dims().len().saturating_sub(2))?.reshape((n, ()))?,
    };

    let seq_len = logits_2d.dim(0)?;
    let greedy = is_greedy(params);
    let mut accepted = Vec::with_capacity(n + 1);

    for i in 0..n {
        if i >= seq_len {
            break;
        }
        let row = logits_2d.i(i)?; // (V,)
        let target_tok = if greedy {
            argmax_token(&row)?
        } else {
            // Build generated context so far for repetition penalty
            let mut ctx = generated.to_vec();
            ctx.extend_from_slice(&accepted);
            sample_token(&row, params, &ctx, rng)?
        };

        if target_tok == draft[i] {
            accepted.push(target_tok);
            stats.accepted += 1;
        } else {
            // Reject rest of draft; keep the *target* token
            accepted.push(target_tok);
            stats.accepted += 1;
            return Ok((accepted, stats));
        }
    }

    // All n draft tokens accepted. Optional bonus: sample one more from last state.
    // Cache already advanced by the chunk forward; sample from last logit row if we
    // only fed n tokens predicting draft[0..n-1]. For a true bonus token we'd need
    // one more forward with draft[n-1]. Do that only when beneficial.
    if n > 0 {
        let last = draft[n - 1];
        let input = Tensor::new(&[last], device)?.unsqueeze(0)?;
        let pos = pos_before_draft + n; // after chunk of n positions
        // Cache length should already reflect chunk; if model advanced by seq_len,
        // pos_before_draft + seq_len is the next position.
        let next_pos = pos_before_draft + seq_len;
        let logits = squeeze_last(&model.forward(&input, next_pos.min(pos), Some(caches))?)?;
        stats.forwards += 1;
        let mut ctx = generated.to_vec();
        ctx.extend_from_slice(&accepted);
        let bonus = if greedy {
            argmax_token(&logits)?
        } else {
            sample_token(&logits, params, &ctx, rng)?
        };
        accepted.push(bonus);
        stats.accepted += 1;
    }

    Ok((accepted, stats))
}

/// Sequential fallback (debug / comparison). N separate forwards — slower.
#[allow(dead_code)]
pub fn verify_and_accept_sequential(
    model: &LoopedMoE,
    caches: &mut RuntimeCaches,
    lens_before_draft: &[usize],
    prev_token: u32,
    pos_before_draft: usize,
    draft: &[u32],
    verify_len: usize,
    device: &Device,
    params: &SamplingParams,
    generated: &[u32],
    rng: &mut impl Rng,
) -> CResult<(Vec<u32>, TeslaStepStats)> {
    caches.restore_lens(lens_before_draft);
    let n = verify_len.min(draft.len());
    let mut stats = TeslaStepStats {
        drafted: draft.len(),
        scheduled: n,
        accepted: 0,
        forwards: 0,
    };
    let mut accepted = Vec::new();
    let mut token_in = prev_token;
    let mut pos = pos_before_draft;

    if n == 0 {
        let input = Tensor::new(&[token_in], device)?.unsqueeze(0)?;
        let logits = squeeze_last(&model.forward(&input, pos, Some(caches))?)?;
        stats.forwards = 1;
        let t = sample_token(&logits, params, generated, rng)?;
        accepted.push(t);
        stats.accepted = 1;
        return Ok((accepted, stats));
    }

    for i in 0..n {
        let input = Tensor::new(&[token_in], device)?.unsqueeze(0)?;
        let logits = squeeze_last(&model.forward(&input, pos, Some(caches))?)?;
        stats.forwards += 1;
        let mut ctx = generated.to_vec();
        ctx.extend_from_slice(&accepted);
        let target_tok = sample_token(&logits, params, &ctx, rng)?;
        if target_tok == draft[i] {
            accepted.push(target_tok);
            token_in = target_tok;
            pos += 1;
            stats.accepted += 1;
        } else {
            accepted.push(target_tok);
            stats.accepted += 1;
            break;
        }
    }
    Ok((accepted, stats))
}
