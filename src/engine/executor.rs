//! Runs forward pass on a batch of sequences

pub struct Executor {
    // TODO: model + kv_cache handles
}

impl Executor {
    pub fn new() -> Self {
        Self {}
    }

    pub fn execute(&mut self, _batch: &[u64]) {
        // TODO: prepare inputs → model.forward → sample
    }
}
