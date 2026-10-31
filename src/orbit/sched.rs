//! OrbitSched — façade over ContinuousScheduler + BlockLedger (O1–O5).

use crate::scheduler::{
    ContinuousScheduler, SchedulerConfig, SchedulerOutputs, Sequence, SequenceStatus,
};
use super::ledger::BlockLedger;

pub struct OrbitSched {
    inner: ContinuousScheduler,
    ledger: Option<BlockLedger>,
}

impl OrbitSched {
    pub fn new(
        cfg: SchedulerConfig,
        num_gpu_blocks: usize,
        block_size: usize,
        ledger: Option<BlockLedger>,
    ) -> Self {
        Self {
            inner: ContinuousScheduler::new(cfg, num_gpu_blocks, block_size),
            ledger,
        }
    }

    pub fn scheduler(&self) -> &ContinuousScheduler {
        &self.inner
    }

    pub fn scheduler_mut(&mut self) -> &mut ContinuousScheduler {
        &mut self.inner
    }

    pub fn ledger(&self) -> Option<&BlockLedger> {
        self.ledger.as_ref()
    }

    pub fn ledger_mut(&mut self) -> Option<&mut BlockLedger> {
        self.ledger.as_mut()
    }

    pub fn add_sequence(&mut self, seq: Sequence) -> u64 {
        let prompt = seq.tokens[..seq.prompt_len.min(seq.tokens.len())].to_vec();
        let id = self.inner.add_sequence(seq);
        if let Some(led) = self.ledger.as_mut() {
            if led.config().enable_prefix_cache && !prompt.is_empty() {
                let _ = led.allocate_with_prefix(id, &prompt);
            } else {
                led.register_seq(id);
                let _ = led.ensure_tokens(id, prompt.len().max(1));
            }
        }
        id
    }

    pub fn schedule(&mut self) -> SchedulerOutputs {
        let out = self.inner.schedule();
        if let Some(led) = self.ledger.as_mut() {
            for &id in out.finished.iter().chain(out.preempted.iter()) {
                led.free_seq(id);
            }
            for &id in out.decode_ids.iter().chain(out.tesla_ids.iter()) {
                if let Some(s) = self.inner.get(id) {
                    let need = s.len() + s.draft_len.unwrap_or(1);
                    let _ = led.ensure_tokens(id, need);
                }
            }
            for chunk in &out.prefill_chunks {
                let need = chunk.start + chunk.len;
                let _ = led.ensure_tokens(chunk.seq_id, need.max(1));
            }
        }
        out
    }

    pub fn has_unfinished(&self) -> bool {
        self.inner.has_unfinished()
    }

    pub fn finish_sequence(&mut self, id: u64, status: SequenceStatus) {
        self.inner.finish_sequence(id, status);
        if let Some(led) = self.ledger.as_mut() {
            led.free_seq(id);
        }
    }

    /// Seal prompt pages into prefix index after full prefill.
    pub fn seal_prefix(&mut self, id: u64, prompt_tokens: &[u32]) {
        if let Some(led) = self.ledger.as_mut() {
            led.seal_seq_prompt(id, prompt_tokens);
        }
    }
}
