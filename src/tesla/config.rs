use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeslaConfig {
    pub enabled: bool,
    /// Max draft tokens (DSpark / DeepInfer next-n, default 5).
    pub max_draft: usize,
    /// Minimum top-1 softmax probability to continue drafting.
    pub min_confidence: f32,
    /// Minimum product of confidences (survival) to schedule verification.
    pub min_survival: f32,
    /// Optional minimum top-2 margin (p1 - p2). 0.0 disables margin check.
    pub min_margin: f32,
    /// Use greedy sampling for the draft model path.
    pub greedy_draft: bool,
    /// Prefer margin-aware scheduling when true.
    pub use_margin: bool,
}

impl Default for TeslaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_draft: 5,
            min_confidence: 0.15,
            min_survival: 0.05,
            min_margin: 0.05,
            greedy_draft: true,
            use_margin: true,
        }
    }
}

impl TeslaConfig {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    pub fn with_nextn(n: usize) -> Self {
        Self {
            max_draft: n.max(1),
            ..Default::default()
        }
    }
}
