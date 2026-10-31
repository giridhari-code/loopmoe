//! Draft paths — **cache-free** so KV only grows on verify (not during draft).

use candle_core::{Device, Result as CResult, Tensor};
use crate::model::LoopedMoE;
use crate::cache::RuntimeCaches;
use crate::sampler::{sample_token, SamplingParams};
use super::confidence::{position_confidence, position_confidence_margin};
use super::config::TeslaConfig;
use super::tree::{draft_tree, TreePath};
use rand::Rng;

pub struct DraftResult {
    pub tokens: Vec<u32>,
    pub confidences: Vec<f32>,
    pub margins: Vec<f32>,
    pub from_tree: bool,
    pub survival: f32,
}

impl From<TreePath> for DraftResult {
    fn from(p: TreePath) -> Self {
        Self {
            tokens: p.tokens,
            confidences: p.confidences,
            margins: p.margins,
            from_tree: true,
            survival: p.survival,
        }
    }
}

fn squeeze(logits: &Tensor) -> CResult<Tensor> {
    match logits.dims() {
        [_] => Ok(logits.clone()),
        [1, _] => logits.squeeze(0),
        [1, 1, _] => logits.squeeze(0)?.squeeze(0),
        d if d.len() == 2 => {
            use candle_core::IndexOp;
            logits.i(0)
        }
        _ => logits.flatten_all(),
    }
}

/// Chain draft **without** touching KV cache (`caches: None`).
/// Token loop is data-dependent, but RAM stays flat — same idea as tree batch path.
/// Real cache is only updated in chunk verify.
pub fn draft_tokens_chain(
    model: &LoopedMoE,
    _caches: &mut RuntimeCaches,
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
    let mut margins = Vec::with_capacity(cfg.max_draft);
    let mut next = start_token;
    let mut pos = start_pos;
    let mut survival = 1.0f32;
    let draft_params = if cfg.greedy_draft {
        SamplingParams::greedy()
    } else {
        params.clone()
    };

    for _ in 0..cfg.max_draft {
        let input = Tensor::new(&[next], device)?.unsqueeze(0)?;
        // No cache — parallel-safe, no KV growth during draft
        let logits = squeeze(&model.forward(&input, pos, None)?)?;

        let c = position_confidence(&logits)?;
        confidences.push(c);
        survival *= c.clamp(1e-6, 1.0);
        if cfg.use_margin {
            margins.push(position_confidence_margin(&logits).unwrap_or(0.0));
        } else {
            margins.push(0.0);
        }

        next = sample_token(&logits, &draft_params, generated, rng)?;
        tokens.push(next);
        pos += 1;
    }

    Ok(DraftResult {
        tokens,
        confidences,
        margins,
        from_tree: false,
        survival,
    })
}

/// Unified entry: tree (batch parallel) or chain (cache-free sequential tokens).
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
    if cfg.use_tree && cfg.tree_width > 1 {
        let path = draft_tree(
            model, caches, start_token, start_pos, device, cfg, params, generated, rng,
        )?;
        Ok(DraftResult::from(path))
    } else {
        draft_tokens_chain(
            model, caches, start_token, start_pos, device, cfg, params, generated, rng,
        )
    }
}
