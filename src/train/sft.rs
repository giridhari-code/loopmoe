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
    let Some(init) = &cfg.init_from else { bail!("sft requires init_from"); };
    let mut model = SentinelTrainModel::new(&cfg.model, device, dtype_for_train())?;
    load_varmap(&model.varmap, Path::new(init), device)?;
    let data = load_lm_jsonl(Path::new(&cfg.data_path), cfg.max_seq_len)?;
    let mut opt = AdamW::new(model.parameters()?, ParamsAdamW { lr: cfg.lr, weight_decay: 0.0, ..Default::default() })?;
    let mut step = 0usize;
    let mut rng = rand::thread_rng();
    let mut order: Vec<usize> = (0..data.len()).collect();
    while step < cfg.max_steps {
        order.shuffle(&mut rng);
        for chunk in order.chunks(cfg.batch_size) {
            let batch: Vec<_> = chunk.iter().map(|&i| data[i].clone()).collect();
            let (input_ids, labels) = collate_lm(&batch, device)?;
            let (_logits, loss) = model.forward(&input_ids, Some(&labels))?;
            let loss = loss.expect("loss");
            let loss_s = loss.to_scalar::<f32>()? as f64;
            opt.backward_step(&loss)?;
            step += 1;
            if step % cfg.log_every == 0 { info!(step, loss = loss_s, "sft"); }
            if step % cfg.save_every == 0 {
                save_varmap(&model.varmap, &Path::new(&cfg.output_dir).join(format!("step-{step}")))?;
            }
            if step >= cfg.max_steps { break; }
        }
    }
    save_varmap(&model.varmap, &Path::new(&cfg.output_dir).join("final"))?;
    info!("sft done");
    Ok(())
}
