//! **Tesla** — Sentinel speculative decoding (DSpark-inspired).
//! DeepSeek name: DSpark. Sentinel name: Tesla.
//!
//! Confidence-scheduled draft → verify with KV rewind.
//! Optimizations: GPU-friendly confidence, top-2 margin, survival product.

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
