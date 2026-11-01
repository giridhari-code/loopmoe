//! Direct Preference Optimization (Rafailov et al.)
//!
//! L = -log σ( β * ( log πθ(y_w) - log πθ(y_l) - log πref(y_w) + log πref(y_l) ) )
//!
//! Policy πθ is trainable; reference πref is frozen (SFT checkpoint).

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::seq::SliceRandom;
use std::path::Path;
use tracing::info;

use super::checkpoint::{dtype_for_train, load_varmap, save_varmap};
use super::config::TrainConfig;
use super::data::{collate_lm, load_dpo_jsonl, LmExample};
use super::model::SentinelTrainModel;

fn ids_as_lm(ids: Vec<u32>) -> LmExample {
    let labels: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    LmExample {
        input_ids: ids,
        labels,
    }
}

/// Softplus(x) = log(1 + exp(x))  — numerically stable via logaddexp-style.
fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
    // log(1 + e^x) = logaddexp(0, x)
    let zeros = x.zeros_like()?;
    let m = x.maximum(&zeros)?;
    let e1 = (x - &m)?.exp()?;
    let e0 = (zeros - &m)?.exp()?;
    (m + (e0 + e1)?.log()?)?
}

pub fn run(cfg: &TrainConfig, device: &Device) -> Result<()> {
    let Some(init) = &cfg.init_from else {
        bail!("dpo requires init_from (SFT checkpoint used as both policy init and reference)");
    };

    // ── Policy (trainable) ──────────────────────────────────────────
    let mut policy = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    policy.aux_coef = 0.0; // no aux during DPO
    load_varmap(&mut policy.varmap, Path::new(init), device)?;

    // ── Reference (frozen, same weights) ────────────────────────────
    let mut reference = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    reference.aux_coef = 0.0;
    load_varmap(&mut reference.varmap, Path::new(init), device)?;
    // Reference never receives optimizer steps — stays frozen.

    let data = load_dpo_jsonl(Path::new(&cfg.data_path), cfg.max_seq_len)?;
    let mut opt = AdamW::new(
        policy.parameters()?,
        ParamsAdamW {
            lr: cfg.lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;

    let beta = cfg.dpo_beta;
    let accum = cfg.grad_accum.max(1);
    let mut step = 0usize;
    let mut micro = 0usize;
    let mut accum_loss = 0.0f64;
    let mut rng = rand::thread_rng();
    let mut order: Vec<usize> = (0..data.len()).collect();

    info!(
        beta,
        ref_ckpt = %init,
        n_pairs = data.len(),
        "DPO start (policy + frozen reference)"
    );

    while step < cfg.max_steps {
        order.shuffle(&mut rng);
        for chunk in order.chunks(cfg.batch_size) {
            let chosen: Vec<LmExample> = chunk
                .iter()
                .map(|&i| ids_as_lm(data[i].chosen_ids.clone()))
                .collect();
            let rejected: Vec<LmExample> = chunk
                .iter()
                .map(|&i| ids_as_lm(data[i].rejected_ids.clone()))
                .collect();

            let (c_ids, c_lab) = collate_lm(&chosen, device)?;
            let (r_ids, r_lab) = collate_lm(&rejected, device)?;

            // Policy log-probs (gradients flow)
            let log_π_w = policy.sequence_logprob(&c_ids, &c_lab)?;
            let log_π_l = policy.sequence_logprob(&r_ids, &r_lab)?;

            // Reference log-probs (no grad — separate model, not in optimizer)
            let log_πref_w = reference.sequence_logprob(&c_ids, &c_lab)?;
            let log_πref_l = reference.sequence_logprob(&r_ids, &r_lab)?;

            // Δ = [log πθ(y_w) - log πθ(y_l)] - [log πref(y_w) - log πref(y_l)]
            let policy_diff = (&log_π_w - &log_π_l)?;
            let ref_diff = (&log_πref_w - &log_πref_l)?;
            let delta = ((&policy_diff - &ref_diff)? * beta)?;

            // L = -log σ(β Δ) = softplus(-β Δ)
            let loss = softplus(&delta.neg()?)?;
            let scaled = (&loss / accum as f64)?;
            let loss_s = scaled.to_scalar::<f32>()? as f64;
            accum_loss += loss_s;

            opt.backward_step(&scaled)?;
            micro += 1;

            if micro % accum == 0 {
                step += 1;
                if step % cfg.log_every == 0 {
                    // Also log implicit reward margin for monitoring
                    let margin = policy_diff.to_scalar::<f32>().unwrap_or(0.0) as f64
                        - ref_diff.to_scalar::<f32>().unwrap_or(0.0) as f64;
                    info!(
                        step,
                        loss = accum_loss / accum as f64,
                        margin,
                        "dpo"
                    );
                }
                if step % cfg.save_every == 0 {
                    save_varmap(
                        &policy.varmap,
                        &Path::new(&cfg.output_dir).join(format!("step-{step}")),
                    )?;
                }
                accum_loss = 0.0;
                if step >= cfg.max_steps {
                    break;
                }
            }
        }
        if step >= cfg.max_steps {
            break;
        }
    }

    save_varmap(&policy.varmap, &Path::new(&cfg.output_dir).join("final"))?;
    info!("dpo done");
    Ok(())
}
