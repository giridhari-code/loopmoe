//! Unified error type for the whole runtime

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("Model load failed: {0}")]
    ModelLoad(String),

    #[error("Scheduler error: {0}")]
    Scheduler(String),

    #[error("Engine error: {0}")]
    Engine(String),

    #[error("CUDA / Kernel error: {0}")]
    Kernel(String),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("HTTP / Server error: {0}")]
    Server(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
