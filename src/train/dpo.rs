//! Direct Preference Optimization (Rafailov et al.)
//!
//! L = -log σ( β * ( log πθ(y_w) - log πθ(y_l) - log πref(y_w) + log πref(y_l) ) )

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::optim::ParamsAdamW;
use rand::seq::SliceRandom;
use std::path::Path;
use tracing::info;

use super::checkpoint::{dtype_for_train, load_varmap, save_checkpoint};
use super::config::TrainConfig;
use super::data::{collate_lm, load_dpo_jsonl, LmExample};
use super::model::SentinelTrainModel;
use super::optim::{load_train_state, LrSchedule, ResumableAdamW};

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
    let mut policy = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    policy.aux_coef = 0.0;

    let mut start_step = 0usize;
    let ref_init: String;

    if let Some(resume) = &cfg.resume_from {
        load_varmap(&mut policy.varmap, Path::new(resume), device)?;
        if let Some(st) = load_train_state(Path::new(resume))? {
            start_step = st.step;
            info!(resume = %resume, start_step, "resuming dpo");
        }
        // Reference stays the original SFT init when possible
        ref_init = cfg
            .init_from
            .clone()
            .unwrap_or_else(|| resume.clone());
    } else {
        let Some(init) = &cfg.init_from else {
            bail!("dpo requires init_from or resume_from");
        };
        load_varmap(&mut policy.varmap, Path::new(init), device)?;
        ref_init = init.clone();
    }

    let mut reference = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    reference.aux_coef = 0.0;
    load_varmap(&mut reference.varmap, Path::new(&ref_init), device)?;

    let data = load_dpo_jsonl(Path::new(&cfg.data_path), cfg.max_seq_len)?;
    let mut opt = ResumableAdamW::from_varmap(
        &policy.varmap,
        ParamsAdamW {
            lr: cfg.lr,
            weight_decay: 0.0,
            ..Default::default()
        },
    )?;
    if let Some(resume) = &cfg.resume_from {
        if Path::new(resume).join("optimizer.safetensors").exists() {
            opt.load(Path::new(resume), device)?;
            info!(optim_step_t = opt.step_t, "restored optimizer state");
        }
    }

    let sched = LrSchedule::new(cfg.lr, cfg.warmup_steps, cfg.max_steps)
        .with_min_ratio(cfg.min_lr_ratio);
    let beta = cfg.dpo_beta;
    let accum = cfg.grad_accum.max(1);
    let mut step = start_step;
    let mut rng = rand::thread_rng();
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut cursor = 0usize;

    info!(
        beta,
        ref_ckpt = %ref_init,
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
        let gnorm = opt.backward_step_clipped(&loss, cfg.max_grad_norm)?;

        step += 1;
        if step % cfg.log_every == 0 {
            info!(step, loss = loss_s, lr, gnorm, margin = last_margin, "dpo");
        }
        if step % cfg.save_every == 0 {
            save_checkpoint(
                &Path::new(&cfg.output_dir).join(format!("step-{step}")),
                &policy.varmap,
                &opt,
                step,
                "dpo",
            )?;
        }
    }

    save_checkpoint(
        &Path::new(&cfg.output_dir).join("final"),
        &policy.varmap,
        &opt,
        step,
        "dpo",
    )?;
    info!(step, "dpo done");
    Ok(())
}
