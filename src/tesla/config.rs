use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeslaConfig {
    pub enabled: bool,
    /// Max draft tokens (DSpark / DeepInfer nextn, default 5).
    pub max_draft: usize,
    pub min_confidence: f32,
    pub min_survival: f32,
    pub greedy_draft: bool,
}

impl Default for TeslaConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_draft: 5,
            min_confidence: 0.15,
            min_survival: 0.05,
            greedy_draft: true,
        }
    }
}

impl TeslaConfig {
    pub fn disabled() -> Self {
        Self { enabled: false, ..Default::default() }
    }
    pub fn with_nextn(n: usize) -> Self {
        Self { max_draft: n.max(1), ..Default::default() }
    }
}
