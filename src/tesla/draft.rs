use candle_core::{Device, Result as CResult, Tensor};
use crate::model::LoopedMoE;
use crate::cache::RuntimeCaches;
use crate::sampler::{sample_token, SamplingParams};
use super::confidence::position_confidence;
use super::config::TeslaConfig;
use rand::Rng;

pub struct DraftResult {
    pub tokens: Vec<u32>,
    pub confidences: Vec<f32>,
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

pub fn draft_tokens(
    model: &LoopedMoE,
    caches: &mut RuntimeCaches,
    start_token: u32,
    start_pos: usize,
    device: &Device,
    cfg: &TeslaConfig,
    params: &SamplingParams,
    generated: &[u32],
    rng: &mut impl Rng,
) -> CResult<DraftResult> {
    let mut tokens = Vec::with_capacity(cfg.max_draft);
    let mut confidences = Vec::with_capacity(cfg.max_draft);
    let mut next = start_token;
    let mut pos = start_pos;
    let draft_params = if cfg.greedy_draft { SamplingParams::greedy() } else { params.clone() };
    for _ in 0..cfg.max_draft {
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        let logits = squeeze(&model.forward(&input, pos, Some(caches))?)?;
        confidences.push(position_confidence(&logits)?);
        next = sample_token(&logits, &draft_params, generated, rng)?;
        tokens.push(next);
        pos += 1;
    }
    Ok(DraftResult { tokens, confidences })
}
