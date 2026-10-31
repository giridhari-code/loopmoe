//! Continuous batching scheduler
//!
//! Decides which sequences to run in the next forward pass,
//! respecting max_num_seqs and max_num_batched_tokens.

use super::sequence::Sequence;

pub struct ContinuousScheduler {
    // TODO: waiting / running / swapped queues
}

impl ContinuousScheduler {
    pub fn new() -> Self {
        Self {}
    }

    pub fn schedule(&mut self) -> Vec<&Sequence> {
        // TODO: implement continuous batching logic
        vec![]
    }
}
