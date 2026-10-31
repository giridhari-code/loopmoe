//! top-k, top-p, temperature, repetition penalty

pub struct Sampler {
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: f32,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: None,
            top_p: Some(0.9),
            repetition_penalty: 1.0,
        }
    }
}

impl Sampler {
    pub fn sample(&self, _logits: &[f32]) -> u32 {
        // TODO: apply temperature / top-k / top-p / rep penalty
        0
    }
}
