//! Core execution engine
//!
//! - `inference` : single-request InferenceEngine (Looped-MoE + Tesla)
//! - `batch`     : continuous scheduler + multi-request BatchEngine
//! - `executor`  : placeholder continuous-batch forward runner
//! - `sampler` / `stop_criteria` / `metrics`

pub mod inference;
pub mod batch;
pub mod executor;
pub mod sampler;
pub mod stop_criteria;
pub mod metrics;

pub use inference::InferenceEngine;
pub use batch::BatchEngine;
pub use executor::Executor;
pub use sampler::Sampler;
pub use stop_criteria::StopCriteria;
pub use metrics::Metrics;
