//! OrbitSched — thin façade over ContinuousScheduler for Orbit core.
//!
//! Keeps existing continuous batching logic; exposes Orbit naming and
//! hooks for BlockLedger (O1) and Tesla lane budgets.

use crate::scheduler::{
    ContinuousScheduler, SchedulerConfig, SchedulerOutputs, Sequence, SequenceStatus,
};
use super::ledger::BlockLedger;

/// Orbit scheduler: time manager + optional ledger sync.
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
        let id = self.inner.add_sequence(seq);
        if let Some(led) = self.ledger.as_mut() {
            led.register_seq(id);
            // Reserve pages for prompt length if known
            if let Some(s) = self.inner.get(id) {
                let n = s.prompt_len.max(1);
                let _ = led.ensure_tokens(id, n);
            }
        }
        id
    }

    pub fn schedule(&mut self) -> SchedulerOutputs {
        let out = self.inner.schedule();
        // Sync ledger free for finished ids (tokens already in finished_buf)
        if let Some(led) = self.ledger.as_mut() {
            for &id in &out.finished {
                led.free_seq(id);
            }
            for &id in &out.preempted {
                led.free_seq(id);
            }
            // Ensure decode growth
            for &id in &out.decode_ids {
                if let Some(s) = self.inner.get(id) {
                    let need = s.len() + s.draft_len.unwrap_or(1);
                    let _ = led.ensure_tokens(id, need);
                }
            }
            for &id in &out.prefill_ids {
                if let Some(s) = self.inner.get(id) {
                    let _ = led.ensure_tokens(id, s.prompt_len.max(1));
                }
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
}
