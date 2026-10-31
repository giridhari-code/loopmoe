//! Safetensors weight discovery + VarBuilder helpers.
//!
//! Supports:
//! - single file: `model.safetensors` or any `*.safetensors`
//! - HF sharded: `model.safetensors.index.json` + `model-00001-of-000XX.safetensors`
//! - directory with multiple `*.safetensors` (sorted)

use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Result as CandleResult};
use candle_nn::VarBuilder;

#[derive(Debug, Clone)]
pub struct WeightPaths {
    pub files: Vec<PathBuf>,
    pub source: WeightSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSource {
    SingleFile,
    ShardedIndex,
    DirectoryGlob,
}

pub struct WeightLoader;

impl WeightLoader {
    /// Resolve safetensor file list from a path (file or directory).
    pub fn discover(path: impl AsRef<Path>) -> anyhow::Result<WeightPaths> {
        let path = path.as_ref();
        if path.is_file() {
            if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e == "safetensors")
            {
                return Ok(WeightPaths {
                    files: vec![path.to_path_buf()],
                    source: WeightSource::SingleFile,
                });
            }
            anyhow::bail!("not a .safetensors file: {}", path.display());
        }

        if !path.is_dir() {
            anyhow::bail!("model path not found: {}", path.display());
        }

        // HF index
        let index = path.join("model.safetensors.index.json");
        if index.exists() {
            let files = Self::from_index(&index, path)?;
            if !files.is_empty() {
                return Ok(WeightPaths {
                    files,
                    source: WeightSource::ShardedIndex,
                });
            }
        }

        // Common single names
        for name in ["model.safetensors", "pytorch_model.safetensors"] {
            let f = path.join(name);
            if f.exists() {
                return Ok(WeightPaths {
                    files: vec![f],
                    source: WeightSource::SingleFile,
                });
            }
        }

        // Glob all shards
        let mut files: Vec<PathBuf> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e == "safetensors")
            })
            .collect();
        files.sort();
        if files.is_empty() {
            anyhow::bail!(
                "no .safetensors under {} (expected model.safetensors or shards)",
                path.display()
            );
        }
        Ok(WeightPaths {
            files,
            source: WeightSource::DirectoryGlob,
        })
    }

    fn from_index(index_path: &Path, dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
        let data = std::fs::read_to_string(index_path)?;
        let v: serde_json::Value = serde_json::from_str(&data)?;
        let weight_map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow::anyhow!("index.json missing weight_map"))?;

        let mut set = std::collections::BTreeSet::new();
        for file in weight_map.values() {
            if let Some(name) = file.as_str() {
                set.insert(dir.join(name));
            }
        }
        Ok(set.into_iter().collect())
    }

    /// Memory-map weights into a Candle VarBuilder.
    pub fn var_builder(
        path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<VarBuilder<'static>> {
        let discovered = Self::discover(path)?;
        tracing::info!(
            n_files = discovered.files.len(),
            source = ?discovered.source,
            first = %discovered.files[0].display(),
            "Loading safetensors"
        );
        for f in &discovered.files {
            if !f.exists() {
                anyhow::bail!("shard missing: {}", f.display());
            }
        }
        // Safety: candle mmap API is unsafe; paths validated above.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&discovered.files, dtype, device)? };
        Ok(vb)
    }

    /// Convenience: load + return paths for logging.
    pub fn load_vb(
        path: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<(VarBuilder<'static>, WeightPaths)> {
        let paths = Self::discover(&path)?;
        let vb = Self::var_builder(path, dtype, device)?;
        Ok((vb, paths))
    }
}

/// Optional dtype override from string ("bf16", "f16", "f32").
pub fn parse_dtype(s: &str) -> CandleResult<DType> {
    match s.to_lowercase().as_str() {
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "f16" | "fp16" | "half" => Ok(DType::F16),
        "f32" | "fp32" | "float" => Ok(DType::F32),
        other => candle_core::bail!("unknown dtype {other}"),
    }
}
