use candle_core::{DType, Result as CResult, Tensor};

pub fn position_confidence(logits: &Tensor) -> CResult<f32> {
    let logits = logits.to_dtype(DType::F32)?;
    let flat = if logits.dims().len() > 1 { logits.flatten_all()? } else { logits };
    let v = flat.to_vec1::<f32>()?;
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    let mut max_e = 0f32;
    for &x in &v {
        let e = (x - max).exp();
        sum += e;
        if (x - max).abs() < 1e-6 { max_e = e; }
    }
    Ok(if sum > 0.0 { max_e / sum } else { 0.0 })
}

pub fn schedule_verify_len(
    confidences: &[f32],
    max_draft: usize,
    min_confidence: f32,
    min_survival: f32,
) -> usize {
    if confidences.is_empty() || max_draft == 0 { return 0; }
    let mut survival = 1.0f32;
    let mut len = 0usize;
    for (i, &c) in confidences.iter().enumerate().take(max_draft) {
        if c < min_confidence { break; }
        survival *= c.clamp(1e-6, 1.0);
        if survival < min_survival { break; }
        len = i + 1;
    }
    len
}
