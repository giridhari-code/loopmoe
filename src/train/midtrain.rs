use anyhow::{bail, Result};
use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW};
use rand::seq::SliceRandom;
use std::path::Path;
use tracing::info;
use super::checkpoint::{dtype_for_train, load_varmap, save_varmap};
use super::config::TrainConfig;
use super::data::{collate_lm, load_lm_jsonl};
use super::model::SentinelTrainModel;

pub fn run(cfg: &TrainConfig, device: &Device) -> Result<()> {
    let Some(init) = &cfg.init_from else { bail!("midtrain requires init_from"); };
    let mut model = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    model.aux_coef = cfg.moe_aux_coef;
    load_varmap(&mut model.varmap, Path::new(init), device)?;
    let data = load_lm_jsonl(Path::new(&cfg.data_path), cfg.max_seq_len)?;
    let mut opt = AdamW::new(
        model.parameters()?,
        ParamsAdamW {
            lr: cfg.lr,
            weight_decay: cfg.weight_decay,
            ..Default::default()
        },
    )?;
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
            opt.backward_step(&scaled)?;
            micro += 1;

            if micro % accum == 0 {
                step += 1;
                if step % cfg.log_every == 0 {
                    info!(step, loss = accum_loss / accum as f64, "midtrain");
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
            }
        }
        if step >= cfg.max_steps {
            break;
        }
    }
    save_varmap(&model.varmap, &Path::new(&cfg.output_dir).join("final"))?;
    info!("midtrain done");
    Ok(())
}
