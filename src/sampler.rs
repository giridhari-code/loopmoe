//! Sampling: greedy / temperature / top-k / top-p.
use candle_core::{DType, Result as CandleResult, Tensor, D};
use rand::Rng;
use std::cmp::Ordering;

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_k: Option<usize>,
    pub top_p: Option<f64>,
    pub repetition_penalty: f64,
    pub stop_token_ids: Vec<u32>,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: None,
            repetition_penalty: 1.0,
            stop_token_ids: Vec::new(),
            seed: None,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self { temperature: 0.0, ..Default::default() }
    }
}

/// Descending f32 compare; NaN treated as Less so sort is total.
#[inline]
fn cmp_f32_desc(a: f32, b: f32) -> Ordering {
    match b.partial_cmp(&a) {
        Some(o) => o,
        None => Ordering::Less,
    }
}

pub fn sample_token(
    logits: &Tensor,
    params: &SamplingParams,
    generated: &[u32],
    rng: &mut impl Rng,
) -> CandleResult<u32> {
    let mut logits = logits.to_dtype(DType::F32)?;
    if logits.dims().len() > 1 {
        logits = logits.flatten_all()?;
    }
    if (params.repetition_penalty - 1.0).abs() > 1e-6 && !generated.is_empty() {
        let mut v = logits.to_vec1::<f32>()?;
        for &id in generated {
            let i = id as usize;
            if i < v.len() {
                if v[i] > 0.0 {
                    v[i] /= params.repetition_penalty as f32;
                } else {
                    v[i] *= params.repetition_penalty as f32;
                }
            }
        }
        logits = Tensor::from_vec(v, logits.dims(), logits.device())?;
    }
    if params.temperature <= 0.0 {
        return logits.argmax(D::Minus1)?.to_scalar::<u32>();
    }
    let mut v = logits.to_vec1::<f32>()?;
    let n = v.len();
    if let Some(k) = params.top_k {
        if k > 0 && k < n {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.select_nth_unstable_by(k, |&a, &b| cmp_f32_desc(v[a], v[b]));
            let thresh = v[idx[k - 1]];
            for x in v.iter_mut() {
                if *x < thresh {
                    *x = f32::NEG_INFINITY;
                }
            }
        }
    }
    let t = params.temperature as f32;
    for x in v.iter_mut() {
        *x /= t;
    }
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for x in v.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    if sum > 0.0 {
        for x in v.iter_mut() {
            *x /= sum;
        }
    }
    if let Some(p) = params.top_p {
        if p > 0.0 && p < 1.0 {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&a, &b| cmp_f32_desc(v[a], v[b]));
            let mut cum = 0f32;
            let mut cutoff = n;
            for (rank, &i) in idx.iter().enumerate() {
                cum += v[i];
                if cum >= p as f32 {
                    cutoff = rank + 1;
                    break;
                }
            }
            let keep: std::collections::HashSet<usize> = idx[..cutoff].iter().copied().collect();
            let mut new_sum = 0f32;
            for (i, x) in v.iter_mut().enumerate() {
                if !keep.contains(&i) {
                    *x = 0.0;
                } else {
                    new_sum += *x;
                }
            }
            if new_sum > 0.0 {
                for x in v.iter_mut() {
                    *x /= new_sum;
                }
            }
        }
    }
    let r: f32 = rng.gen();
    let mut cum = 0f32;
    for (i, &p) in v.iter().enumerate() {
        cum += p;
        if r <= cum {
            return Ok(i as u32);
        }
    }
    Ok((n - 1) as u32)
}

pub fn is_stop(token: u32, params: &SamplingParams) -> bool {
    params.stop_token_ids.contains(&token)
}
