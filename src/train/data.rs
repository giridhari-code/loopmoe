use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Deserialize)]
struct LmRow {
    input_ids: Vec<u32>,
    #[serde(default)]
    labels: Option<Vec<i64>>,
}

#[derive(Debug, Deserialize)]
struct DpoRow {
    chosen_ids: Vec<u32>,
    rejected_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct LmExample {
    pub input_ids: Vec<u32>,
    pub labels: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct DpoExample {
    pub chosen_ids: Vec<u32>,
    pub rejected_ids: Vec<u32>,
}

pub fn load_lm_jsonl(path: &Path, max_seq: usize) -> Result<Vec<LmExample>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let row: LmRow = serde_json::from_str(&line).with_context(|| format!("line {}", i + 1))?;
        let mut ids = row.input_ids;
        if ids.len() > max_seq { ids.truncate(max_seq); }
        if ids.len() < 2 { continue; }
        let labels = match row.labels {
            Some(mut lab) => {
                lab.truncate(ids.len());
                while lab.len() < ids.len() { lab.push(-100); }
                lab
            }
            None => ids.iter().map(|&x| x as i64).collect(),
        };
        out.push(LmExample { input_ids: ids, labels });
    }
    if out.is_empty() { bail!("no examples in {}", path.display()); }
    Ok(out)
}

pub fn load_dpo_jsonl(path: &Path, max_seq: usize) -> Result<Vec<DpoExample>> {
    let f = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() { continue; }
        let mut row: DpoRow = serde_json::from_str(&line).with_context(|| format!("dpo line {}", i + 1))?;
        row.chosen_ids.truncate(max_seq);
        row.rejected_ids.truncate(max_seq);
        if row.chosen_ids.len() < 2 || row.rejected_ids.len() < 2 { continue; }
        out.push(DpoExample { chosen_ids: row.chosen_ids, rejected_ids: row.rejected_ids });
    }
    if out.is_empty() { bail!("no dpo examples"); }
    Ok(out)
}

pub fn collate_lm(batch: &[LmExample], device: &Device) -> Result<(Tensor, Tensor)> {
    let max_len = batch.iter().map(|b| b.input_ids.len()).max().unwrap_or(1);
    let bsz = batch.len();
    let mut ids = vec![0u32; bsz * max_len];
    let mut labels = vec![-100i64; bsz * max_len];
    for (i, ex) in batch.iter().enumerate() {
        let n = ex.input_ids.len();
        ids[i * max_len..i * max_len + n].copy_from_slice(&ex.input_ids);
        for (j, &lab) in ex.labels.iter().enumerate().take(n) {
            labels[i * max_len + j] = lab;
        }
    }
    Ok((
        Tensor::from_vec(ids, (bsz, max_len), device)?,
        Tensor::from_vec(labels, (bsz, max_len), device)?,
    ))
}
