use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarMap;
use std::path::Path;

use super::optim::{save_train_state, TrainState, ResumableAdamW};

pub fn save_varmap(varmap: &VarMap, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    varmap
        .save(dir.join("model.safetensors"))
        .context("save safetensors")?;
    Ok(())
}

pub fn load_varmap(varmap: &mut VarMap, dir: &Path, _device: &Device) -> Result<()> {
    let path = if dir.is_file() {
        dir.to_path_buf()
    } else {
        dir.join("model.safetensors")
    };
    varmap.load(&path).context("load safetensors")?;
    Ok(())
}

/// Save model weights + optimizer moments + train step.
pub fn save_checkpoint(
    dir: &Path,
    varmap: &VarMap,
    opt: &ResumableAdamW,
    step: usize,
    stage: &str,
) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    save_varmap(varmap, dir)?;
    opt.save(dir)?;
    save_train_state(
        dir,
        &TrainState {
            step,
            stage: stage.to_string(),
        },
    )?;
    Ok(())
}

pub fn dtype_for_train() -> DType {
    DType::F32
}
