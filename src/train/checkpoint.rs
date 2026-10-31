use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarMap;
use std::path::Path;

pub fn save_varmap(varmap: &VarMap, dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    varmap.save(dir.join("model.safetensors")).context("save safetensors")?;
    Ok(())
}

pub fn load_varmap(varmap: &mut VarMap, dir: &Path, _device: &Device) -> Result<()> {
    let path = if dir.is_file() { dir.to_path_buf() } else { dir.join("model.safetensors") };
    varmap.load(&path).context("load safetensors")?;
    Ok(())
}

pub fn dtype_for_train() -> DType {
    DType::F32
}
