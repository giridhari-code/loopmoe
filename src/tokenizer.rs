//! HF tokenizer.json + byte-level fallback.
use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer as HfTokenizer;

pub struct Tokenizer { inner: Backend }

enum Backend { Hf(HfTokenizer), ByteLevel }

impl Tokenizer {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = if path.is_dir() {
            let j = path.join("tokenizer.json");
            if !j.exists() { bail!("tokenizer.json not found in {}", path.display()); }
            j
        } else { path.to_path_buf() };
        let hf = HfTokenizer::from_file(&file).map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;
        Ok(Self { inner: Backend::Hf(hf) })
    }

    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let j = dir.as_ref().join("tokenizer.json");
        if j.exists() { return Self::from_path(j); }
        tracing::warn!("no tokenizer.json — byte-level fallback");
        Ok(Self::byte_level())
    }

    pub fn byte_level() -> Self { Self { inner: Backend::ByteLevel } }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        match &self.inner {
            Backend::Hf(hf) => {
                let enc = hf.encode(text, true).map_err(|e| anyhow::anyhow!("encode: {e}"))?;
                Ok(enc.get_ids().to_vec())
            }
            Backend::ByteLevel => Ok(text.bytes().map(|b| b as u32).collect()),
        }
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match &self.inner {
            Backend::Hf(hf) => hf.decode(ids, true).map_err(|e| anyhow::anyhow!("decode: {e}")),
            Backend::ByteLevel => {
                let bytes: Vec<u8> = ids.iter().filter_map(|&i| u8::try_from(i).ok()).collect();
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
        }
    }

    pub fn eos_id(&self) -> Option<u32> {
        match &self.inner {
            Backend::Hf(hf) => hf.token_to_id("</s>")
                .or_else(|| hf.token_to_id("<|endoftext|>"))
                .or_else(|| hf.token_to_id("<eos>")),
            Backend::ByteLevel => None,
        }
    }
}

pub fn resolve_tokenizer_path(model_path: &Path, explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit { return p.to_path_buf(); }
    if model_path.is_dir() { return model_path.join("tokenizer.json"); }
    model_path.parent().unwrap_or_else(|| Path::new(".")).join("tokenizer.json")
}
