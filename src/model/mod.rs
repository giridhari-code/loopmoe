//! Model architectures + weight loading (inference core).
//!
//! **Production path:** `LoopedMoE` (Sentinel looped MoE + optional MLA).
//! **Loader:** multi-file / HF-index safetensors.
//! **Trait:** `InferModel` for Orbit / engine abstraction.

mod attention;
mod layers;
mod moe;
mod mla;
mod looped_moe;
pub mod rope;
pub mod loader;
pub mod traits;

// Future dense ports (not production yet)
pub mod llama;
pub mod qwen;
pub mod mlp;
pub mod quantization;

pub use attention::*;
pub use layers::*;
pub use moe::*;
pub use mla::*;
pub use looped_moe::LoopedMoE;
pub use rope::RotaryEmbedding;
pub use loader::{parse_dtype, WeightLoader, WeightPaths, WeightSource};
pub use traits::InferModel;

pub type SentinelV1 = LoopedMoE;
