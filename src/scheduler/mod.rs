//! Continuous batching scheduler + paged KV-cache management
//!
//! Aligned with DeepInfer engine scheduler concepts.

pub mod continuous;
pub mod sequence;
pub mod block_manager;

pub use continuous::{ContinuousScheduler, SchedulerConfig, SchedulerOutputs};
pub use sequence::{Sequence, SequenceStatus, SequenceState};
pub use block_manager::{AllocStatus, BlockManager};
