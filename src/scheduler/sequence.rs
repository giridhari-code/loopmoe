//! Sequence = one generation request

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceState {
    Waiting,
    Running,
    Swapped,
    Finished,
}

#[derive(Debug)]
pub struct Sequence {
    pub id: u64,
    pub tokens: Vec<u32>,
    pub state: SequenceState,
    pub num_computed_tokens: usize,
    // TODO: block table, sampling params, etc.
}

impl Sequence {
    pub fn new(id: u64, prompt_tokens: Vec<u32>) -> Self {
        Self {
            id,
            tokens: prompt_tokens,
            state: SequenceState::Waiting,
            num_computed_tokens: 0,
        }
    }
}
