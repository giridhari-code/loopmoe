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
use super::optim::{backward_step_clipped, LrSchedule};

fn ids_as_lm(ids: Vec<u32>) -> LmExample {
    let labels: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
    LmExample {
        input_ids: ids,
        labels,
    }
}

fn softplus(x: &Tensor) -> candle_core::Result<Tensor> {
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

    let mut policy = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    policy.aux_coef = 0.0;
    load_varmap(&mut policy.varmap, Path::new(init), device)?;

    let mut reference = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    reference.aux_coef = 0.0;
    load_varmap(&mut reference.varmap, Path::new(init), device)?;

    let data = load_dpo_jsonl(Path::new(&cfg.data_path), cfg.max_seq_len)?;
    let vars = policy.parameters()?;
    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW {
            lr: cfg.lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    let sched = LrSchedule::new(cfg.lr, cfg.warmup_steps, cfg.max_steps)
        .with_min_ratio(cfg.min_lr_ratio);
    let beta = cfg.dpo_beta;
    let accum = cfg.grad_accum.max(1);
    let mut step = 0usize;
    let mut rng = rand::thread_rng();
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut cursor = 0usize;

    info!(
        beta,
        ref_ckpt = %init,
        n_pairs = data.len(),
        "DPO start (policy + frozen reference)"
    );

    while step < cfg.max_steps {
        let mut total: Option<Tensor> = None;
        let mut n_micro = 0usize;
        let mut last_margin = 0.0f64;

        for _ in 0..accum {
            if cursor >= order.len() {
                order.shuffle(&mut rng);
                cursor = 0;
            }
            let end = (cursor + cfg.batch_size).min(order.len());
            let idxs: Vec<usize> = order[cursor..end].to_vec();
            cursor = end;
            if idxs.is_empty() {
                continue;
            }

            let chosen: Vec<LmExample> = idxs
                .iter()
                .map(|&i| ids_as_lm(data[i].chosen_ids.clone()))
                .collect();
            let rejected: Vec<LmExample> = idxs
                .iter()
                .map(|&i| ids_as_lm(data[i].rejected_ids.clone()))
                .collect();

            let (c_ids, c_lab) = collate_lm(&chosen, device)?;
            let (r_ids, r_lab) = collate_lm(&rejected, device)?;

            let log_π_w = policy.sequence_logprob(&c_ids, &c_lab)?;
            let log_π_l = policy.sequence_logprob(&r_ids, &r_lab)?;
            let log_πref_w = reference.sequence_logprob(&c_ids, &c_lab)?;
            let log_πref_l = reference.sequence_logprob(&r_ids, &r_lab)?;

            let policy_diff = (&log_π_w - &log_π_l)?;
            let ref_diff = (&log_πref_w - &log_πref_l)?;
            let delta = ((&policy_diff - &ref_diff)? * beta)?;
            let loss = softplus(&delta.neg()?)?;

            last_margin = policy_diff.to_scalar::<f32>().unwrap_or(0.0) as f64
                - ref_diff.to_scalar::<f32>().unwrap_or(0.0) as f64;

            total = Some(match total {
                None => loss,
                Some(t) => (t + loss)?,
            });
            n_micro += 1;
        }
        if n_micro == 0 {
            continue;
        }

        let loss = (total.unwrap() / n_micro as f64)?;
        let loss_s = loss.to_scalar::<f32>()? as f64;
        let lr = sched.lr_at(step);
        opt.set_learning_rate(lr);
        let gnorm = backward_step_clipped(&mut opt, &loss, &vars, cfg.max_grad_norm)?;

        step += 1;
        if step % cfg.log_every == 0 {
            info!(step, loss = loss_s, lr, gnorm, margin = last_margin, "dpo");
        }
        if step % cfg.save_every == 0 {
            save_varmap(
                &policy.varmap,
                &Path::new(&cfg.output_dir).join(format!("step-{step}")),
            )?;
        }
    }

    save_varmap(&policy.varmap, &Path::new(&cfg.output_dir).join("final"))?;
    info!("dpo done");
    Ok(())
}
