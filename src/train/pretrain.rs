use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::optim::ParamsAdamW;
use rand::seq::SliceRandom;
use std::path::Path;
use tracing::info;

use super::checkpoint::{dtype_for_train, load_varmap, save_checkpoint};
use super::config::TrainConfig;
use super::data::{collate_lm, load_lm_jsonl};
use super::model::SentinelTrainModel;
use super::optim::{load_train_state, LrSchedule, ResumableAdamW};

pub fn run(cfg: &TrainConfig, device: &Device) -> Result<()> {
    info!(stage = "pretrain", data = %cfg.data_path, "start");
    let mut model = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    model.aux_coef = cfg.moe_aux_coef;

    let mut start_step = 0usize;
    if let Some(resume) = &cfg.resume_from {
        load_varmap(&mut model.varmap, Path::new(resume), device)?;
        if let Some(st) = load_train_state(Path::new(resume))? {
            start_step = st.step;
            info!(resume = %resume, start_step, "resuming pretrain");
        }
    } else if let Some(init) = &cfg.init_from {
        load_varmap(&mut model.varmap, Path::new(init), device)?;
    }

    let data = load_lm_jsonl(Path::new(&cfg.data_path), cfg.max_seq_len)?;
    let mut opt = ResumableAdamW::from_varmap(
        &model.varmap,
        ParamsAdamW {
            lr: cfg.lr,
            weight_decay: cfg.weight_decay,
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
    let accum = cfg.grad_accum.max(1);
    let mut step = start_step;
    let mut rng = rand::thread_rng();
    let mut order: Vec<usize> = (0..data.len()).collect();
    let mut cursor = 0usize;

    while step < cfg.max_steps {
        let mut total: Option<Tensor> = None;
        let mut n_micro = 0usize;
        for _ in 0..accum {
            if cursor >= order.len() {
                order.shuffle(&mut rng);
                cursor = 0;
            }
            let end = (cursor + cfg.batch_size).min(order.len());
            let batch: Vec<_> = order[cursor..end].iter().map(|&i| data[i].clone()).collect();
            cursor = end;
            if batch.is_empty() {
                continue;
            }
            let (input_ids, labels) = collate_lm(&batch, device)?;
            let (_logits, loss) = model.forward(&input_ids, Some(&labels))?;
            let loss = loss.expect("loss");
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
            info!(step, loss = loss_s, lr, gnorm, "pretrain");
        }
        if step % cfg.save_every == 0 {
            save_checkpoint(
                &Path::new(&cfg.output_dir).join(format!("step-{step}")),
                &model.varmap,
                &opt,
                step,
                "pretrain",
            )?;
        }
    }
    save_checkpoint(
        &Path::new(&cfg.output_dir).join("final"),
        &model.varmap,
        &opt,
        step,
        "pretrain",
    )?;
    info!(step, "pretrain done");
    Ok(())
}
