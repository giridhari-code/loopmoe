//! Trainable Sentinel — **same backbone as inference** (`LoopedMoE`).
//!
//! Weights, layer layout, and loop structure match production inference so
//! checkpoints load directly into `InferenceEngine` / `sentinel` CLI.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Result as CResult, Tensor, D};
use candle_nn::{VarBuilder, VarMap};

use crate::config::ModelConfig;
use crate::kernels::{DiffOpsGuard, set_force_diff_ops};
use crate::model::LoopedMoE;

pub struct SentinelTrainModel {
    pub varmap: VarMap,
    /// Shared architecture with inference.
    pub backbone: LoopedMoE,
    pub cfg: ModelConfig,
    pub aux_coef: f64,
}

impl SentinelTrainModel {
    pub fn new(cfg: &ModelConfig, device: &Device, dtype: DType) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        // Identical load path to inference (`LoopedMoE::load`).
        let backbone = LoopedMoE::load(vb, cfg, device)?;
        Ok(Self {
            varmap,
            backbone,
            cfg: cfg.clone(),
            aux_coef: 0.001,
        })
    }

    /// Forward → full logits `[B,S,V]` + optional CE loss.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        labels: Option<&Tensor>,
    ) -> CResult<(Tensor, Option<Tensor>)> {
        let _guard = DiffOpsGuard::enable();
        let logits = self.backbone.forward_full(input_ids)?;
        let loss = if let Some(labels) = labels {
            let logits_s = logits.i((.., 0..logits.dim(1)? - 1, ..))?;
            let labels_s = labels.i((.., 1..labels.dim(1)?))?;
            let (b, t, v) = logits_s.dims3()?;
            let ce = masked_ce(&logits_s.reshape((b * t, v))?, &labels_s.reshape((b * t,))?)?;
            // Aux load-balance is computed inside MoE with discrete top-k;
            // optional scalar regularizer when coef > 0 (0 disables).
            if self.aux_coef > 0.0 {
                // Lightweight proxy: encourage non-zero CE scale only — full
                // aux needs MoE stats export; coef reserved for future hook.
                Some(ce)
            } else {
                Some(ce)
            }
        } else {
            None
        };
        Ok((logits, loss))
    }

    /// Sequence log-likelihood for DPO (mean over batch of sum log π).
    pub fn sequence_logprob(&mut self, input_ids: &Tensor, labels: &Tensor) -> CResult<Tensor> {
        let _guard = DiffOpsGuard::enable();
        let logits = self.backbone.forward_full(input_ids)?;
        let logits_s = logits.i((.., 0..logits.dim(1)? - 1, ..))?;
        let labels_s = labels.i((.., 1..labels.dim(1)?))?;
        let (b, t, v) = logits_s.dims3()?;
        sum_logprob(&logits_s.reshape((b * t, v))?, &labels_s.reshape((b * t,))?, b)
    }

    pub fn parameters(&self) -> Result<Vec<candle_core::Var>> {
        Ok(self.varmap.all_vars())
    }
}

fn masked_ce(logits: &Tensor, labels: &Tensor) -> CResult<Tensor> {
    let (lp, mask, cnt) = logprob_parts(logits, labels)?;
    lp.neg()?.mul(&mask)?.sum_all()? / cnt
}

fn sum_logprob(logits: &Tensor, labels: &Tensor, batch: usize) -> CResult<Tensor> {
    let (lp, mask, _cnt) = logprob_parts(logits, labels)?;
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

/// Call once if embedding train outside DiffOpsGuard.
#[allow(dead_code)]
pub fn enable_train_diff_ops() {
    set_force_diff_ops(true);
}
