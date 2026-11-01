use anyhow::Result;
use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::seq::SliceRandom;
use std::path::Path;
use tracing::info;

use super::checkpoint::{dtype_for_train, load_varmap, save_varmap};
use super::config::TrainConfig;
use super::data::{collate_lm, load_lm_jsonl};
use super::model::SentinelTrainModel;
use super::optim::{backward_step_clipped, LrSchedule};

pub fn run(cfg: &TrainConfig, device: &Device) -> Result<()> {
    info!(stage = "pretrain", data = %cfg.data_path, "start");
    let mut model = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    model.aux_coef = cfg.moe_aux_coef;
    if let Some(init) = &cfg.init_from {
        load_varmap(&mut model.varmap, Path::new(init), device)?;
    }
    let data = load_lm_jsonl(Path::new(&cfg.data_path), cfg.max_seq_len)?;
    let vars = model.parameters()?;
    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW {
            lr: cfg.lr,
            weight_decay: cfg.weight_decay,
            ..Default::default()
        },
    )?;
    let sched = LrSchedule::new(cfg.lr, cfg.warmup_steps, cfg.max_steps)
        .with_min_ratio(cfg.min_lr_ratio);
    let accum = cfg.grad_accum.max(1);
    let mut step = 0usize;
    let mut micro = 0usize;
    let mut accum_loss = 0.0f64;
    let mut rng = rand::thread_rng();
    let mut order: Vec<usize> = (0..data.len()).collect();

    while step < cfg.max_steps {
        order.shuffle(&mut rng);
        for chunk in order.chunks(cfg.batch_size) {
            let batch: Vec<_> = chunk.iter().map(|&i| data[i].clone()).collect();
            let (input_ids, labels) = collate_lm(&batch, device)?;
            let (_logits, loss) = model.forward(&input_ids, Some(&labels))?;
            let loss = loss.expect("loss");
            let scaled = (loss / accum as f64)?;
            let loss_s = scaled.to_scalar::<f32>()? as f64;
            accum_loss += loss_s;

            // Accumulate grads; only clip+step on boundary
            let grads = scaled.backward()?;
            micro += 1;

            if micro % accum == 0 {
                let lr = sched.lr_at(step);
                opt.set_learning_rate(lr);
                let gnorm = if cfg.max_grad_norm > 0.0 {
                    super::optim::clip_grad_norm(&vars, cfg.max_grad_norm)?
                } else {
                    0.0
                };
                opt.step(&grads)?;

                step += 1;
                if step % cfg.log_every == 0 {
                    info!(
                        step,
                        loss = accum_loss / accum as f64,
                        lr,
                        gnorm,
                        "pretrain"
                    );
                }
                if step % cfg.save_every == 0 {
                    save_varmap(
                        &model.varmap,
                        &Path::new(&cfg.output_dir).join(format!("step-{step}")),
                    )?;
                }
                accum_loss = 0.0;
                if step >= cfg.max_steps {
                    break;
                }
            } else {
                // micro-batch: just accumulate (candle accumulates into Var grads)
                let _ = grads;
                let _ = backward_step_clipped; // keep import used if needed
            }
        }
        if step >= cfg.max_steps {
            break;
        }
    }
    save_varmap(&model.varmap, &Path::new(&cfg.output_dir).join("final"))?;
    info!("pretrain done");
    Ok(())
}
