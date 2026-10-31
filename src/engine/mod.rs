//! Core execution engine
//!
//! - `inference`  : current production InferenceEngine (Looped-MoE + Tesla)
//! - `executor`   : future continuous-batching forward runner
//! - `sampler`    : top-k / top-p / temperature (new path)
//! - `stop_criteria` / `metrics`

pub mod inference;
pub mod executor;
pub mod sampler;
pub mod stop_criteria;
pub mod metrics;

pub use inference::InferenceEngine;
pub use executor::Executor;
pub use sampler::Sampler;
pub use stop_criteria::StopCriteria;
pub use metrics::Metrics;
