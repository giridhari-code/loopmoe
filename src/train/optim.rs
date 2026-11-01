//! Shared training optim helpers: LR schedule, grad clip, resumable AdamW.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Result as CResult, Tensor, Var};
use candle_nn::optim::ParamsAdamW;
use serde::{Deserialize, Serialize};

// ─── LR schedule ────────────────────────────────────────────────────────────

/// Linear warmup → cosine decay to `min_lr_ratio * peak_lr`.
#[derive(Debug, Clone)]
pub struct LrSchedule {
    pub peak_lr: f64,
    pub warmup_steps: usize,
    pub total_steps: usize,
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

// ─── Grad clip ──────────────────────────────────────────────────────────────

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

// ─── Resumable AdamW ────────────────────────────────────────────────────────

/// Per-parameter Adam moments + global step. Matches `candle_nn::AdamW` update
/// (decoupled weight decay + bias correction) but exposes state for checkpointing.
pub struct ResumableAdamW {
    vars: Vec<(String, Var)>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    pub params: ParamsAdamW,
    /// Bias-correction step counter (t). Must be restored on resume.
    pub step_t: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimMeta {
    pub step_t: usize,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl ResumableAdamW {
    pub fn new(named_vars: Vec<(String, Var)>, params: ParamsAdamW) -> Result<Self> {
        let mut m = Vec::with_capacity(named_vars.len());
        let mut v = Vec::with_capacity(named_vars.len());
        for (_, var) in &named_vars {
            let t = var.as_tensor();
            m.push(t.zeros_like().context("adam m zeros")?);
            v.push(t.zeros_like().context("adam v zeros")?);
        }
        Ok(Self {
            vars: named_vars,
            m,
            v,
            params,
            step_t: 0,
        })
    }

    pub fn from_varmap(varmap: &candle_nn::VarMap, params: ParamsAdamW) -> Result<Self> {
        let data = varmap.data().lock().map_err(|_| anyhow::anyhow!("varmap lock"))?;
        let mut named: Vec<(String, Var)> = data
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        // Stable order for save/load by name (already keyed); sort for determinism.
        named.sort_by(|a, b| a.0.cmp(&b.0));
        Self::new(named, params)
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    pub fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    pub fn vars(&self) -> Vec<Var> {
        self.vars.iter().map(|(_, v)| v.clone()).collect()
    }

    /// AdamW step using grads already attached to Vars (after backward + optional clip).
    pub fn step_from_var_grads(&mut self) -> Result<()> {
        self.step_t += 1;
        let t = self.step_t as f64;
        let p = &self.params;
        let lr = p.lr;
        let b1 = p.beta1;
        let b2 = p.beta2;
        let eps = p.eps;
        let wd = p.weight_decay;
        let bc1 = 1.0 - b1.powf(t);
        let bc2 = 1.0 - b2.powf(t);

        for i in 0..self.vars.len() {
            let var = &self.vars[i].1;
            let Ok(g) = var.grad() else {
                continue;
            };
            let g = g.to_dtype(DType::F32)?;
            // m ← β1 m + (1-β1) g
            self.m[i] = ((&self.m[i] * b1)? + (&g * (1.0 - b1))?)?;
            // v ← β2 v + (1-β2) g²
            let g2 = g.sqr()?;
            self.v[i] = ((&self.v[i] * b2)? + (&g2 * (1.0 - b2))?)?;
            let m_hat = (&self.m[i] / bc1)?;
            let v_hat = (&self.v[i] / bc2)?;
            let denom = (v_hat.sqrt()? + eps)?;
            let update = (&m_hat / &denom)?;
            // decoupled weight decay: θ ← θ(1 - lr·λ) - lr · update
            let theta = var.as_tensor().to_dtype(DType::F32)?;
            let decayed = (&theta * (1.0 - lr * wd))?;
            let new_theta = (decayed - (&update * lr)?)?;
            var.set(&new_theta.to_dtype(var.dtype())?)?;
        }
        Ok(())
    }

    pub fn backward_step_clipped(
        &mut self,
        loss: &Tensor,
        max_norm: f64,
    ) -> Result<f64> {
        let _grads = loss.backward()?;
        let vars = self.vars();
        let norm = if max_norm > 0.0 {
            clip_grad_norm(&vars, max_norm)?
        } else {
            grad_norm(&vars).unwrap_or(0.0)
        };
        self.step_from_var_grads()?;
        Ok(norm)
    }

    /// Save moments to `optimizer.safetensors` + `optim_meta.json`.
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (i, (name, _)) in self.vars.iter().enumerate() {
            tensors.insert(format!("{name}__m"), self.m[i].copy()?);
            tensors.insert(format!("{name}__v"), self.v[i].copy()?);
        }
        candle_core::safetensors::save(&tensors, dir.join("optimizer.safetensors"))
            .context("save optimizer.safetensors")?;
        let meta = OptimMeta {
            step_t: self.step_t,
            lr: self.params.lr,
            beta1: self.params.beta1,
            beta2: self.params.beta2,
            eps: self.params.eps,
            weight_decay: self.params.weight_decay,
        };
        std::fs::write(
            dir.join("optim_meta.json"),
            serde_json::to_string_pretty(&meta)?,
        )?;
        Ok(())
    }

    /// Restore moments + step_t. Parameter names must match.
    pub fn load(&mut self, dir: &Path, device: &Device) -> Result<()> {
        let path = dir.join("optimizer.safetensors");
        if !path.exists() {
            bail!("optimizer.safetensors not found in {}", dir.display());
        }
        let loaded = candle_core::safetensors::load(&path, device)
            .context("load optimizer.safetensors")?;
        for (i, (name, var)) in self.vars.iter().enumerate() {
            let mk = format!("{name}__m");
            let vk = format!("{name}__v");
            let m = loaded
                .get(&mk)
                .with_context(|| format!("missing moment {mk}"))?
                .to_device(device)?
                .to_dtype(DType::F32)?;
            let v = loaded
                .get(&vk)
                .with_context(|| format!("missing moment {vk}"))?
                .to_device(device)?
                .to_dtype(DType::F32)?;
            if m.shape() != var.shape() || v.shape() != var.shape() {
                bail!("moment shape mismatch for {name}");
            }
            self.m[i] = m;
            self.v[i] = v;
        }
        let meta_path = dir.join("optim_meta.json");
        if meta_path.exists() {
            let meta: OptimMeta = serde_json::from_str(&std::fs::read_to_string(meta_path)?)?;
            self.step_t = meta.step_t;
            // Keep current LR schedule control; restore hyperparams except lr
            self.params.beta1 = meta.beta1;
            self.params.beta2 = meta.beta2;
            self.params.eps = meta.eps;
            self.params.weight_decay = meta.weight_decay;
        }
        Ok(())
    }
}

// ─── Train run state (step counter) ─────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainState {
    pub step: usize,
    pub stage: String,
}

pub fn save_train_state(dir: &Path, state: &TrainState) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(
        dir.join("train_state.json"),
        serde_json::to_string_pretty(state)?,
    )?;
    Ok(())
}

pub fn load_train_state(dir: &Path) -> Result<Option<TrainState>> {
    let p = dir.join("train_state.json");
    if !p.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&std::fs::read_to_string(p)?)?))
}

/// Legacy helper kept for callers that still use candle AdamW + GradMap path.
pub fn backward_step_clipped(
    opt: &mut candle_nn::AdamW,
    loss: &Tensor,
    vars: &[Var],
    max_norm: f64,
) -> CResult<f64> {
    use candle_nn::Optimizer;
    let grads = loss.backward()?;
    let norm = if max_norm > 0.0 {
        clip_grad_norm(vars, max_norm)?
    } else {
        0.0
    };
    opt.step(&grads)?;
    Ok(norm)
}
