//! Model architectures + weight loading

pub mod loader;
pub mod llama;
pub mod qwen;
pub mod attention;
pub mod mlp;
pub mod rope;
pub mod quantization;

// Existing specialized models kept for now
// pub mod looped_moe; etc. live under the old paths
