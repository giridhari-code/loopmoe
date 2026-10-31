use candle_core::{Device, Result as CResult, Tensor};
use crate::model::LoopedMoE;
use crate::cache::RuntimeCaches;
use crate::sampler::{sample_token, SamplingParams};
use rand::Rng;

#[derive(Debug, Default, Clone)]
pub struct TeslaStepStats {
    pub drafted: usize,
    pub scheduled: usize,
    pub accepted: usize,
}

fn squeeze(logits: &Tensor) -> CResult<Tensor> {
    match logits.dims() {
        [_] => Ok(logits.clone()),
        [1, _] => logits.squeeze(0),
        [1, 1, _] => logits.squeeze(0)?.squeeze(0),
        d if d.len() == 2 => { use candle_core::IndexOp; logits.i(0) }
        _ => logits.flatten_all(),
    }
}

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
    let mut stats = TeslaStepStats { drafted: draft.len(), scheduled: n, accepted: 0 };
    let mut accepted = Vec::new();
    let mut token_in = prev_token;
    let mut pos = pos_before_draft;
    if n == 0 {
        let input = Tensor::new(&[token_in], device)?.unsqueeze(0)?;
        let logits = squeeze(&model.forward(&input, pos, Some(caches))?)?;
        let t = sample_token(&logits, params, generated, rng)?;
        accepted.push(t);
        stats.accepted = 1;
        return Ok((accepted, stats));
    }
    for i in 0..n {
        let input = Tensor::new(&[token_in], device)?.unsqueeze(0)?;
        let logits = squeeze(&model.forward(&input, pos, Some(caches))?)?;
        let target_tok = sample_token(&logits, params, generated, rng)?;
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
