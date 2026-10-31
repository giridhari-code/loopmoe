//! Tree draft — **parallel** level-synchronous expand.
//!
//! At each depth all beam nodes advance in **one batched forward** (batch matmul),
//! not sequential per-node KV replay.

use candle_core::{Device, IndexOp, Result as CResult, Tensor};
use crate::model::LoopedMoE;
use crate::cache::RuntimeCaches;
use crate::sampler::SamplingParams;
use super::confidence::{position_confidence, position_confidence_margin};
use super::config::TeslaConfig;
use super::fuse::topk_last_ids;
use rand::Rng;
use rayon::prelude::*;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct TreePath {
    pub tokens: Vec<u32>,
    pub confidences: Vec<f32>,
    pub margins: Vec<f32>,
    pub survival: f32,
}

#[derive(Debug, Clone)]
struct BeamNode {
    tokens: Vec<u32>,
    confidences: Vec<f32>,
    margins: Vec<f32>,
    survival: f32,
    last_token: u32,
}

#[inline]
fn cmp_survival_desc(a: f32, b: f32) -> Ordering {
    match b.partial_cmp(&a) {
        Some(o) => o,
        None => Ordering::Less,
    }
}

/// Parallel tree draft: batch all beam last-tokens → one forward → branch top-w.
/// Draft scoring uses `caches: None` so the batch path is pure parallel GEMM/attention
/// without sequential KV restore/replay. Verify still uses real KV.
pub fn draft_tree(
    model: &LoopedMoE,
    _caches: &mut RuntimeCaches,
    start_token: u32,
    start_pos: usize,
    device: &Device,
    cfg: &TeslaConfig,
    _params: &SamplingParams,
    _generated: &[u32],
    _rng: &mut impl Rng,
) -> CResult<TreePath> {
    let depth = cfg.tree_depth_or_default();
    let width = cfg.tree_width.max(1);

    let mut beam = vec![BeamNode {
        tokens: Vec::new(),
        confidences: Vec::new(),
        margins: Vec::new(),
        survival: 1.0,
        last_token: start_token,
    }];

    for d in 0..depth {
        let b = beam.len();
        if b == 0 {
            break;
        }

        // Parallel batch forward: (B, 1) tokens in one call
        let batch_toks: Vec<u32> = beam.iter().map(|n| n.last_token).collect();
        let input = Tensor::new(batch_toks.as_slice(), device)?.reshape((b, 1))?;
        let logits = model.forward(&input, start_pos + d, None)?;

        let logits_2d = match logits.dims() {
            [bb, _v] if *bb == b => logits.clone(),
            [1, bb, _v] if *bb == b => logits.squeeze(0)?,
            [bb, 1, _v] if *bb == b => logits.squeeze(1)?,
            _ => logits.flatten_all()?.reshape((b, ()))?,
        };

        let width = width;
        let use_margin = cfg.use_margin;
        let min_conf = cfg.min_confidence;
        let min_surv = cfg.min_survival;

        let row_results: Vec<CResult<Vec<BeamNode>>> = (0..b)
            .into_par_iter()
            .map(|i| {
                let node = &beam[i];
                let row = logits_2d.i(i)?;
                let conf = position_confidence(&row).unwrap_or(0.0);
                let margin = if use_margin {
                    position_confidence_margin(&row).unwrap_or(0.0)
                } else {
                    0.0
                };

                let row_b = row.unsqueeze(0)?;
                let (_vals, ids) = topk_last_ids(&row_b, width)?;
                let id_tensor = ids.squeeze(0)?;
                let id_vec = id_tensor.to_vec1::<u32>().or_else(|_| {
                    id_tensor
                        .to_dtype(candle_core::DType::U32)?
                        .to_vec1::<u32>()
                })?;

                let mut kids = Vec::with_capacity(id_vec.len());
                for &cand in &id_vec {
                    let survival = node.survival * conf.clamp(1e-6, 1.0);
                    if conf < min_conf && !node.tokens.is_empty() {
                        continue;
                    }
                    if survival < min_surv && !node.tokens.is_empty() {
                        continue;
                    }
                    let mut tokens = node.tokens.clone();
                    tokens.push(cand);
                    let mut confidences = node.confidences.clone();
                    confidences.push(conf);
                    let mut margins = node.margins.clone();
                    margins.push(margin);
                    kids.push(BeamNode {
                        tokens,
                        confidences,
                        margins,
                        survival,
                        last_token: cand,
                    });
                }
                Ok(kids)
            })
            .collect();

        let mut next_beam = Vec::new();
        for r in row_results {
            next_beam.extend(r?);
        }

        if next_beam.is_empty() {
            break;
        }

        next_beam.par_sort_by(|a, b| cmp_survival_desc(a.survival, b.survival));
        let keep = (width * 2).max(width);
        next_beam.truncate(keep);
        beam = next_beam;
    }

    beam.par_sort_by(|a, b| cmp_survival_desc(a.survival, b.survival));

    if let Some(best) = beam.into_iter().next() {
        Ok(TreePath {
            tokens: best.tokens,
            confidences: best.confidences,
            margins: best.margins,
            survival: best.survival,
        })
    } else {
        Ok(TreePath {
            tokens: Vec::new(),
            confidences: Vec::new(),
            margins: Vec::new(),
            survival: 0.0,
        })
    }
}
