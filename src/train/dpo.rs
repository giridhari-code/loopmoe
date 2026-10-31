use anyhow::{bail, Result};
use candle_core::Device;
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
    LmExample { input_ids: ids, labels }
}

pub fn run(cfg: &TrainConfig, device: &Device) -> Result<()> {
    let Some(init) = &cfg.init_from else { bail!("dpo requires init_from"); };
    let mut policy = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    load_varmap(&mut policy.varmap, Path::new(init), device)?;
    let data = load_dpo_jsonl(Path::new(&cfg.data_path), cfg.max_seq_len)?;
    let mut opt = AdamW::new(policy.parameters()?, ParamsAdamW { lr: cfg.lr, weight_decay: 0.0, ..Default::default() })?;
    let beta = cfg.dpo_beta;
    let mut step = 0usize;
    let mut rng = rand::thread_rng();
    let mut order: Vec<usize> = (0..data.len()).collect();
    while step < cfg.max_steps {
        order.shuffle(&mut rng);
        for chunk in order.chunks(cfg.batch_size) {
            let chosen: Vec<LmExample> = chunk.iter().map(|&i| ids_as_lm(data[i].chosen_ids.clone())).collect();
            let rejected: Vec<LmExample> = chunk.iter().map(|&i| ids_as_lm(data[i].rejected_ids.clone())).collect();
            let (c_ids, c_lab) = collate_lm(&chosen, device)?;
            let (r_ids, r_lab) = collate_lm(&rejected, device)?;
            let (_, loss_c) = policy.forward(&c_ids, Some(&c_lab))?;
            let (_, loss_r) = policy.forward(&r_ids, Some(&r_lab))?;
            let delta = ((&loss_c.expect("c") - &loss_r.expect("r"))? * beta)?;
            let loss = (delta.exp()? + 1.0)?.log()?; // softplus
            let loss_s = loss.to_scalar::<f32>()? as f64;
            opt.backward_step(&loss)?;
            step += 1;
            if step % cfg.log_every == 0 { info!(step, loss = loss_s, "dpo"); }
            if step % cfg.save_every == 0 {
                save_varmap(&policy.varmap, &Path::new(&cfg.output_dir).join(format!("step-{step}")))?;
            }
            if step >= cfg.max_steps { break; }
        }
    }
    save_varmap(&policy.varmap, &Path::new(&cfg.output_dir).join("final"))?;
    info!("dpo done");
    Ok(())
}
