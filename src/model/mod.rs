//! Model architectures + weight loading
//!
//! Existing Sentinel modules + new general-purpose placeholders (Llama/Qwen path).

// ── Existing specialized model ──────────────────────────────────────────────
mod attention;
mod layers;
mod moe;
mod mla;
mod looped_moe;
pub mod rope;

pub use attention::*;
pub use layers::*;
pub use moe::*;
pub use mla::*;
pub use looped_moe::LoopedMoE;
pub use rope::RotaryEmbedding;

pub type SentinelV1 = LoopedMoE;

// ── New general architecture placeholders (for future multi-model support) ──
pub mod loader;
pub mod llama;
pub mod qwen;
pub mod mlp;
pub mod quantization;
// note: attention.rs and rope.rs already exist above; the new ones are
// intentionally not dual-declared to avoid name conflicts.
