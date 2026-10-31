use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeslaConfig {
    pub enabled: bool,
    /// Max draft tokens on the chain path (also max tree depth when tree is on).
    pub max_draft: usize,
    /// Minimum top-1 softmax probability to continue drafting.
    pub min_confidence: f32,
    /// Minimum product of confidences (survival) to schedule verification.
    pub min_survival: f32,
    /// Optional minimum top-2 margin (p1 - p2). 0.0 disables margin check.
    pub min_margin: f32,
    /// Use greedy sampling for the linear draft path.
    pub greedy_draft: bool,
    /// Prefer margin-aware scheduling when true.
    pub use_margin: bool,
    /// Enable tree (branched) draft — parallel level expand.
    pub use_tree: bool,
    /// Branching factor at each tree node (top-w tokens).
    pub tree_width: usize,
    /// Max depth of the draft tree (defaults to max_draft if 0).
    pub tree_depth: usize,
    /// Use DoBu fused GEMM helpers inside Tesla paths when possible.
    pub use_dobu_fuse: bool,
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
            // Parallel-first defaults
            use_tree: true,
            tree_width: 2,
            tree_depth: 0,
            use_dobu_fuse: true,
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
            use_tree: true,
            tree_depth: n.max(1),
            ..Default::default()
        }
    }

    pub fn with_tree(width: usize, depth: usize) -> Self {
        Self {
            use_tree: true,
            tree_width: width.max(1),
            tree_depth: depth,
            max_draft: depth.max(1),
            ..Default::default()
        }
    }

    pub fn tree_depth_or_default(&self) -> usize {
        if self.tree_depth == 0 {
            self.max_draft.max(1)
        } else {
            self.tree_depth
        }
    }
}
