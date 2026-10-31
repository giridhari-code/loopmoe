//! Tree draft — branched speculative candidates ranked by survival.
//!
//! At each depth we expand the top-`width` tokens from the current node,
//! keep a beam of paths, and select the best path (highest survival product)
//! for chunk verify.

use candle_core::{Device, IndexOp, Result as CResult, Tensor, D};
use crate::model::LoopedMoE;
use crate::cache::RuntimeCaches;
use crate::sampler::{sample_token, SamplingParams};
use super::confidence::{position_confidence, position_confidence_margin};
use super::config::TeslaConfig;
use super::fuse::topk_last_ids;
use rand::Rng;

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
    /// Last token fed to the model for this node.
    last_token: u32,
    pos: usize,
}

fn squeeze(logits: &Tensor) -> CResult<Tensor> {
    match logits.dims() {
        [_] => Ok(logits.clone()),
        [1, _] => logits.squeeze(0),
        [1, 1, _] => logits.squeeze(0)?.squeeze(0),
        d if d.len() == 2 => logits.i(0),
        _ => logits.flatten_all(),
    }
}

/// Expand a draft tree and return the best path by survival product.
pub fn draft_tree(
    model: &LoopedMoE,
    caches: &mut RuntimeCaches,
    start_token: u32,
    start_pos: usize,
    device: &Device,
    cfg: &TeslaConfig,
    params: &SamplingParams,
    generated: &[u32],
    rng: &mut impl Rng,
) -> CResult<TreePath> {
    let depth = cfg.tree_depth_or_default();
    let width = cfg.tree_width.max(1);
    let lens_root = caches.checkpoint_lens();

    // Root beam: one node at the current position (no tokens yet).
    let mut beam = vec![BeamNode {
        tokens: Vec::new(),
        confidences: Vec::new(),
        margins: Vec::new(),
        survival: 1.0,
        last_token: start_token,
        pos: start_pos,
    }];

    let draft_params = if cfg.greedy_draft {
        SamplingParams::greedy()
    } else {
        params.clone()
    };

    for _d in 0..depth {
        let mut next_beam: Vec<BeamNode> = Vec::new();

        for node in &beam {
            // Restore to root then replay this path so each branch has correct KV.
            caches.restore_lens(&lens_root);
            let mut pos = start_pos;
            let mut tok = start_token;
            for &t in &node.tokens {
                let input = Tensor::new(&[tok], device)?.unsqueeze(0)?;
                let _ = model.forward(&input, pos, Some(caches))?;
                tok = t;
                pos += 1;
            }
            // Forward from node's last state
            let input = Tensor::new(&[node.last_token], device)?.unsqueeze(0)?;
            let logits = squeeze(&model.forward(&input, node.pos, Some(caches))?)?;

            let conf = position_confidence(&logits).unwrap_or(0.0);
            let margin = if cfg.use_margin {
                position_confidence_margin(&logits).unwrap_or(0.0)
            } else {
                0.0
            };

            // Branch: top-w tokens
            let k = width.min(logits.dims1().unwrap_or(width).max(1));
            let (_vals, ids) = topk_last_ids(&logits.unsqueeze(0)?, k)?;
            // ids shape (1, k)
            let id_vec = ids.squeeze(0)?.to_vec1::<u32>().or_else(|_| {
                ids.squeeze(0)?.to_dtype(candle_core::DType::U32)?.to_vec1::<u32>()
            })?;

            for &cand in &id_vec {
                // If not greedy, optionally mix in a sample for diversity
                let chosen = if cfg.greedy_draft {
                    cand
                } else if id_vec.len() == 1 {
                    sample_token(&logits, &draft_params, generated, rng)?
                } else {
                    cand
                };

                let mut tokens = node.tokens.clone();
                tokens.push(chosen);
                let mut confidences = node.confidences.clone();
                confidences.push(conf);
                let mut margins = node.margins.clone();
                margins.push(margin);
                let survival = node.survival * conf.clamp(1e-6, 1.0);

                if conf < cfg.min_confidence || survival < cfg.min_survival {
                    // Still keep short paths; ranking will prefer better ones
                }

                next_beam.push(BeamNode {
                    tokens,
                    confidences,
                    margins,
                    survival,
                    last_token: chosen,
                    pos: node.pos + 1,
                });
            }
        }

        if next_beam.is_empty() {
            break;
        }

        // Keep top beams by survival (beam width = width for memory control)
        next_beam.sort_by(|a, b| {
            b.survival
                .partial_cmp(&a.survival)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        next_beam.truncate(width.max(1) * 2); // allow slight expansion then prune
        beam = next_beam;
    }

    // Restore root cache for caller (verify will restore again itself)
    caches.restore_lens(&lens_root);

    // Best path
    beam.sort_by(|a, b| {
        b.survival
            .partial_cmp(&a.survival)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

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
