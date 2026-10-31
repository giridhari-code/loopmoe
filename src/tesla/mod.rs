//! **Tesla** — Sentinel speculative decoding (DSpark-inspired).
//! DeepSeek name: DSpark. Sentinel name: Tesla.
//!
//! Confidence-scheduled draft → **chunk verify** (one sequence forward =
//! heavy matrix multiplications once) → KV rewind on reject.
//!
//! Optimizations:
//! - GPU-friendly confidence + top-2 margin
//! - Survival-product + margin-aware schedule
//! - Chunk verify (O(1) forwards per step vs O(k))
//! - Greedy argmax fast-path

pub mod config;
pub mod confidence;
pub mod draft;
pub mod verify;

pub use config::TeslaConfig;
pub use confidence::{
    position_confidence, position_confidence_margin, schedule_verify_len,
    schedule_verify_len_smart,
};
pub use draft::draft_tokens;
pub use verify::{verify_and_accept, TeslaStepStats};
