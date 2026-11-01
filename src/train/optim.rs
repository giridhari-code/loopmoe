//! Shared training optim helpers: LR schedule + gradient clipping.

use candle_core::{Result as CResult, Tensor, Var};

/// Linear warmup → cosine decay to `min_lr_ratio * peak_lr`.
#[derive(Debug, Clone)]
pub struct LrSchedule {
    pub peak_lr: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
    /// Floor as fraction of peak (default 0.1).
    pub min_lr_ratio: f64,
}

impl LrSchedule {
    pub fn new(peak_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            peak_lr,
            warmup_steps,
            total_steps: total_steps.max(1),
            min_lr_ratio: 0.1,
        }
    }

    pub fn with_min_ratio(mut self, ratio: f64) -> Self {
        self.min_lr_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// LR at optimizer step `step` (0-indexed, after each accum boundary).
    pub fn lr_at(&self, step: usize) -> f64 {
        let peak = self.peak_lr;
        let min_lr = peak * self.min_lr_ratio;
        if self.warmup_steps > 0 && step < self.warmup_steps {
            return peak * (step as f64 + 1.0) / self.warmup_steps as f64;
        }
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let t = (step.saturating_sub(self.warmup_steps) as f64 / decay_steps as f64).min(1.0);
        min_lr + 0.5 * (peak - min_lr) * (1.0 + (std::f64::consts::PI * t).cos())
    }
}

/// Global L2 grad-norm. Returns ‖g‖₂ over all parameters that have a grad.
pub fn grad_norm(vars: &[Var]) -> CResult<f64> {
    let mut total_sq = 0.0f64;
    for v in vars {
        if let Ok(g) = v.grad() {
            let sq = g.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
            total_sq += sq;
        }
    }
    Ok(total_sq.sqrt())
}

/// Clip grads in-place so ‖g‖₂ ≤ `max_norm`. Returns the pre-clip norm.
///
/// Call **after** `loss.backward()` and **before** `optimizer.step(&grads)`.
/// Relies on `Var::set_grad` sharing storage with the `GradMap` from backward.
pub fn clip_grad_norm(vars: &[Var], max_norm: f64) -> CResult<f64> {
    let norm = grad_norm(vars)?;
    if max_norm <= 0.0 || norm <= max_norm || norm <= 0.0 {
        return Ok(norm);
    }
    let scale = max_norm / (norm + 1e-6);
    for v in vars {
        if let Ok(g) = v.grad() {
            let scaled = (g * scale)?;
            v.set_grad(&scaled)?;
        }
    }
    Ok(norm)
}

/// One clipped optimizer step: backward → clip → AdamW step.
/// Returns (pre-clip grad norm).
pub fn backward_step_clipped(
    opt: &mut candle_nn::AdamW,
    loss: &Tensor,
    vars: &[Var],
    max_norm: f64,
) -> CResult<f64> {
    let grads = loss.backward()?;
    let norm = if max_norm > 0.0 {
        clip_grad_norm(vars, max_norm)?
    } else {
        0.0
    };
    opt.step(&grads)?;
    Ok(norm)
}
