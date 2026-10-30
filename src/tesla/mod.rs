//! **Tesla** — Sentinel speculative decoding (DSpark-inspired).
//! DeepSeek name: DSpark. Sentinel name: Tesla.

pub mod config;
pub mod confidence;
pub mod draft;
pub mod verify;

pub use config::TeslaConfig;
pub use confidence::{position_confidence, schedule_verify_len};
pub use draft::draft_tokens;
pub use verify::{verify_and_accept, TeslaStepStats};
