//! Sequence = one generation request
//!
//! Status model aligned with DeepInfer `SequenceStatus`
//! (csrc/engine/sequence.hpp).

use crate::engine::stop_criteria::StopCriteria;
use crate::sampler::SamplingParams;

/// Mirrors DeepInfer SequenceStatus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Swapped,
    FinishedStopped,
    FinishedLengthCapped,
    FinishedAborted,
    FinishedIgnored,
    FinishedPreempted,
}

impl SequenceStatus {
    pub fn is_finished(self) -> bool {
        matches!(
            self,
            Self::FinishedStopped
                | Self::FinishedLengthCapped
                | Self::FinishedAborted
                | Self::FinishedIgnored
                | Self::FinishedPreempted
        )
    }

    pub fn finish_reason(self) -> Option<&'static str> {
        match self {
            Self::FinishedStopped => Some("stop"),
            Self::FinishedLengthCapped => Some("length"),
            Self::FinishedAborted => Some("abort"),
            Self::FinishedIgnored => Some("prompt_length_exceeded"),
            Self::FinishedPreempted => Some("insufficient_system_resource"),
            _ => None,
        }
    }
}

/// Back-compat alias used by older placeholders.
pub type SequenceState = SequenceStatus;

#[derive(Debug, Clone)]
pub struct Sequence {
    pub id: u64,
    pub request_id: String,
    /// Full token history: prompt + generated.
    pub tokens: Vec<u32>,
    pub prompt_len: usize,
    pub status: SequenceStatus,
    /// Tokens already processed by the model (prefill + decode steps).
    pub num_computed_tokens: usize,
    /// Physical block table (DeepInfer-style). Block ids are 1-based; 0 = unused.
    pub block_table: Vec<u32>,
    pub sampling: SamplingParams,
    pub stop: StopCriteria,
    /// Optional Tesla / DSpark draft length hint for this sequence.
    pub draft_len: Option<usize>,
}

impl Sequence {
    pub fn new(id: u64, prompt_tokens: Vec<u32>) -> Self {
        let prompt_len = prompt_tokens.len();
        Self {
            id,
            request_id: id.to_string(),
            tokens: prompt_tokens,
            prompt_len,
            status: SequenceStatus::Waiting,
            num_computed_tokens: 0,
            block_table: Vec::new(),
            sampling: SamplingParams::default(),
            stop: StopCriteria {
                max_tokens: 256,
                stop_token_ids: Vec::new(),
                stop_strings: Vec::new(),
            },
            draft_len: None,
        }
    }

    pub fn with_request_id(mut self, rid: impl Into<String>) -> Self {
        self.request_id = rid.into();
        self
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_finished(&self) -> bool {
        self.status.is_finished()
    }

    pub fn is_prompt(&self) -> bool {
        self.num_computed_tokens < self.prompt_len
    }

    pub fn num_new_tokens(&self) -> usize {
        self.tokens.len().saturating_sub(self.num_computed_tokens)
    }

    pub fn generated_len(&self) -> usize {
        self.tokens.len().saturating_sub(self.prompt_len)
    }

    pub fn append_token(&mut self, token: u32) {
        self.tokens.push(token);
    }

    pub fn mark_computed(&mut self, n: usize) {
        self.num_computed_tokens = self.num_computed_tokens.saturating_add(n).min(self.tokens.len());
    }

    pub fn finish(&mut self, status: SequenceStatus) {
        debug_assert!(status.is_finished());
        self.status = status;
    }
}
