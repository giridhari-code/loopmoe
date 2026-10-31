//! Tesla confidence estimation — DSpark / speculative decoding.
//! Prefer Candle ops so large-vocab models stay GPU-side as long as possible.

use candle_core::{DType, Result as CResult, Tensor, D};

/// Max-softmax probability (top-1 confidence).
/// Uses argmax + gather so we never materialize the full softmax vector on CPU.
pub fn position_confidence(logits: &Tensor) -> CResult<f32> {
    let logits = logits.to_dtype(DType::F32)?;
    let flat = if logits.dims().len() > 1 {
        logits.flatten_all()?
    } else {
        logits.clone()
    };

    // Stable max + exp-sum via Candle
    let max = flat.max(0)?;
    let max_val = max.to_scalar::<f32>()?;
    let shifted = (&flat - max_val)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_all()?.to_scalar::<f32>()?;
    if sum <= 0.0 {
        return Ok(0.0);
    }
    // max_e / sum  (max_e == 1.0 after shift)
    Ok(1.0 / sum)
}

/// Top-2 margin: (p1 - p2). Higher = more confident.
/// Falls back to top-1 if vocab is tiny.
pub fn position_confidence_margin(logits: &Tensor) -> CResult<f32> {
    let logits = logits.to_dtype(DType::F32)?;
    let flat = if logits.dims().len() > 1 {
        logits.flatten_all()?
    } else {
        logits.clone()
    };
    let n = flat.dims1()?;
    if n < 2 {
        return position_confidence(&flat);
    }

    let max = flat.max(0)?;
    let max_val = max.to_scalar::<f32>()?;
    let shifted = (&flat - max_val)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_all()?.to_scalar::<f32>()?;
    if sum <= 0.0 {
        return Ok(0.0);
    }

    // Approximate top-2 by masking the max and taking new max.
    // (Exact top-k would need sort; this is fast + good enough for scheduling.)
    let p1 = 1.0 / sum;
    // Zero out the max position approximately by subtracting a large value
    // then recompute — for speed we use a soft version.
    let soft_mask = ((&flat - max_val)? * 50.0)?.exp()?; // sharp peak
    let approx_p2_mass = ((&exp - &soft_mask)?).max(0)?.to_scalar::<f32>().unwrap_or(0.0).max(0.0);
    let p2 = (approx_p2_mass / sum).clamp(0.0, 1.0 - p1);
    Ok((p1 - p2).clamp(0.0, 1.0))
}

/// Schedule how many draft tokens to verify.
/// Uses survival product of confidences; stops early on low confidence.
pub fn schedule_verify_len(
    confidences: &[f32],
    max_draft: usize,
    min_confidence: f32,
    min_survival: f32,
) -> usize {
    if confidences.is_empty() || max_draft == 0 {
        return 0;
    }
    let mut survival = 1.0f32;
    let mut len = 0usize;
    for (i, &c) in confidences.iter().enumerate().take(max_draft) {
        if c < min_confidence {
            break;
        }
        survival *= c.clamp(1e-6, 1.0);
        if survival < min_survival {
            break;
        }
        len = i + 1;
    }
    len
}

/// Combined schedule: prefer margin when available, else pure top-1.
pub fn schedule_verify_len_smart(
    confidences: &[f32],
    margins: Option<&[f32]>,
    max_draft: usize,
    min_confidence: f32,
    min_survival: f32,
    min_margin: f32,
) -> usize {
    if confidences.is_empty() || max_draft == 0 {
        return 0;
    }
    let mut survival = 1.0f32;
    let mut len = 0usize;
    for i in 0..max_draft.min(confidences.len()) {
        let c = confidences[i];
        if c < min_confidence {
            break;
        }
        if let Some(m) = margins {
            if i < m.len() && m[i] < min_margin {
                break;
            }
        }
        survival *= c.clamp(1e-6, 1.0);
        if survival < min_survival {
            break;
        }
        len = i + 1;
    }
    len
}
