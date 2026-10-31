//! Continuous batching scheduler — optimized
//!
//! Patterns from DeepInfer `Scheduler`:
//! - waiting / running queues
//! - allocate blocks before admit
//! - prefer decode, then prefill (shortest first)
//! - preempt longest seq when capacity tight

use super::block_manager::{AllocStatus, BlockManager};
use super::sequence::{Sequence, SequenceStatus};

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    /// When true, admit waiting prompts shortest-first (better packing).
    pub shortest_first: bool,
    /// Enable preemption of longest running seq when blocks run out.
    pub enable_preempt: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            shortest_first: true,
            enable_preempt: true,
        }
    }
}

#[derive(Debug, Default)]
pub struct SchedulerOutputs {
    /// Sequences that need a prefill this step.
    pub prefill_ids: Vec<u64>,
    /// Sequences that need a decode this step.
    pub decode_ids: Vec<u64>,
    pub num_batched_tokens: usize,
    /// Already reaped (blocks freed) before this step ran.
    pub finished: Vec<u64>,
    pub preempted: Vec<u64>,
}

impl SchedulerOutputs {
    pub fn is_empty(&self) -> bool {
        self.prefill_ids.is_empty() && self.decode_ids.is_empty()
    }

    pub fn scheduled_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.prefill_ids.iter().chain(self.decode_ids.iter()).copied()
    }

    pub fn num_scheduled(&self) -> usize {
        self.prefill_ids.len() + self.decode_ids.len()
    }
}

pub struct ContinuousScheduler {
    cfg: SchedulerConfig,
    block_manager: BlockManager,
    waiting: Vec<Sequence>,
    running: Vec<Sequence>,
    next_id: u64,
    /// Cumulative stats
    pub stats: SchedulerStats,
}

#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    pub schedule_calls: u64,
    pub total_prefills_admitted: u64,
    pub total_decodes_scheduled: u64,
    pub total_preempts: u64,
    pub total_finished: u64,
}

impl ContinuousScheduler {
    pub fn new(cfg: SchedulerConfig, num_gpu_blocks: usize, block_size: usize) -> Self {
        Self {
            cfg,
            block_manager: BlockManager::new(num_gpu_blocks, block_size),
            waiting: Vec::new(),
            running: Vec::new(),
            next_id: 1,
            stats: SchedulerStats::default(),
        }
    }

    pub fn block_manager(&self) -> &BlockManager {
        &self.block_manager
    }

    pub fn block_manager_mut(&mut self) -> &mut BlockManager {
        &mut self.block_manager
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.cfg
    }

    pub fn add_sequence(&mut self, mut seq: Sequence) -> u64 {
        if seq.id == 0 {
            seq.id = self.next_id;
            self.next_id += 1;
        }
        seq.status = SequenceStatus::Waiting;
        let id = seq.id;
        self.waiting.push(seq);
        id
    }

    pub fn num_unfinished(&self) -> usize {
        self.waiting.len() + self.running.len()
    }

    pub fn has_unfinished(&self) -> bool {
        self.num_unfinished() > 0
    }

    pub fn running_ids(&self) -> Vec<u64> {
        self.running.iter().map(|s| s.id).collect()
    }

    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Decode token budget estimate (Tesla may accept >1 token per step).
    #[inline]
    fn decode_token_estimate(seq: &Sequence) -> usize {
        seq.draft_len.unwrap_or(1).max(1)
    }

    pub fn schedule(&mut self) -> SchedulerOutputs {
        self.stats.schedule_calls += 1;
        let mut out = SchedulerOutputs::default();

        // ── 1. Reap finished ──────────────────────────────────────────────
        let mut still = Vec::with_capacity(self.running.len());
        for seq in self.running.drain(..) {
            if seq.is_finished() {
                out.finished.push(seq.id);
                self.block_manager.free_many(&seq.block_table);
                self.stats.total_finished += 1;
            } else {
                still.push(seq);
            }
        }
        self.running = still;

        let mut num_tokens = 0usize;
        let max_seqs = self.cfg.max_num_seqs;
        let max_tok = self.cfg.max_num_batched_tokens;

        // ── 2. Schedule decode (running, non-prompt) ──────────────────────
        for seq in self.running.iter_mut() {
            if seq.is_prompt() {
                // still in prefill phase of first step after admit — rare
                continue;
            }
            if out.decode_ids.len() + out.prefill_ids.len() >= max_seqs {
                break;
            }
            let est = Self::decode_token_estimate(seq);
            if num_tokens + est > max_tok && !out.decode_ids.is_empty() {
                break;
            }
            // Ensure room for growth
            let need = self
                .block_manager
                .blocks_for_tokens(seq.len() + est)
                .saturating_sub(seq.block_table.len());
            if need > 0 {
                if !self.try_allocate_or_preempt(seq, need, &mut out) {
                    continue;
                }
            }
            out.decode_ids.push(seq.id);
            num_tokens += est;
            self.stats.total_decodes_scheduled += 1;
        }

        // ── 3. Admit prefills (shortest-first optional) ───────────────────
        if self.cfg.shortest_first && self.waiting.len() > 1 {
            self.waiting
                .sort_by_key(|s| s.prompt_len); // ascending
        }

        let mut admitted = Vec::new();
        let mut waiting_keep = Vec::with_capacity(self.waiting.len());

        for mut seq in self.waiting.drain(..) {
            let slots_used = self.running.len() + admitted.len() + out.decode_ids.len();
            if slots_used >= max_seqs {
                waiting_keep.push(seq);
                continue;
            }
            let prompt_tokens = seq.prompt_len.max(1);
            if num_tokens + prompt_tokens > max_tok && !out.decode_ids.is_empty() && admitted.is_empty()
            {
                // Prefer finishing decode-only step; defer long prefills
                waiting_keep.push(seq);
                continue;
            }
            if num_tokens + prompt_tokens > max_tok && !admitted.is_empty() {
                waiting_keep.push(seq);
                continue;
            }

            let need = self.block_manager.blocks_for_tokens(seq.len().max(1));
            match self.block_manager.can_allocate(need) {
                AllocStatus::Ok => {
                    if let Some(blocks) = self.block_manager.allocate(need) {
                        seq.block_table = blocks;
                        seq.status = SequenceStatus::Running;
                        num_tokens += prompt_tokens;
                        out.prefill_ids.push(seq.id);
                        admitted.push(seq);
                        self.stats.total_prefills_admitted += 1;
                    } else {
                        waiting_keep.push(seq);
                    }
                }
                AllocStatus::Later => {
                    // Try preemption once
                    if self.cfg.enable_preempt && self.preempt_one(&mut out) {
                        if let Some(blocks) = self.block_manager.allocate(need) {
                            seq.block_table = blocks;
                            seq.status = SequenceStatus::Running;
                            num_tokens += prompt_tokens;
                            out.prefill_ids.push(seq.id);
                            admitted.push(seq);
                            self.stats.total_prefills_admitted += 1;
                            continue;
                        }
                    }
                    waiting_keep.push(seq);
                }
                AllocStatus::Never => {
                    seq.finish(SequenceStatus::FinishedIgnored);
                    out.finished.push(seq.id);
                    self.stats.total_finished += 1;
                }
            }
        }
        self.waiting = waiting_keep;
        self.running.extend(admitted);

        out.num_batched_tokens = num_tokens;
        out
    }

    /// Allocate `need` blocks for `seq`, preempting if necessary.
    fn try_allocate_or_preempt(
        &mut self,
        seq: &mut Sequence,
        need: usize,
        out: &mut SchedulerOutputs,
    ) -> bool {
        match self.block_manager.can_allocate(need) {
            AllocStatus::Ok => {
                if let Some(extra) = self.block_manager.allocate(need) {
                    seq.block_table.extend(extra);
                    true
                } else {
                    false
                }
            }
            AllocStatus::Later if self.cfg.enable_preempt => {
                // Don't preempt ourselves — only others
                if self.preempt_one_except(seq.id, out) {
                    if let Some(extra) = self.block_manager.allocate(need) {
                        seq.block_table.extend(extra);
                        return true;
                    }
                }
                false
            }
            _ => false,
        }
    }

    /// Preempt the longest running sequence (most KV pressure).
    fn preempt_one(&mut self, out: &mut SchedulerOutputs) -> bool {
        self.preempt_one_except(u64::MAX, out)
    }

    fn preempt_one_except(&mut self, except_id: u64, out: &mut SchedulerOutputs) -> bool {
        let mut best: Option<(usize, usize)> = None; // (idx, len)
        for (i, seq) in self.running.iter().enumerate() {
            if seq.id == except_id || seq.is_finished() {
                continue;
            }
            let len = seq.len();
            if best.map(|(_, l)| len > l).unwrap_or(true) {
                best = Some((i, len));
            }
        }
        let Some((idx, _)) = best else {
            return false;
        };
        let mut seq = self.running.swap_remove(idx);
        // Remove from decode list if present
        out.decode_ids.retain(|&id| id != seq.id);
        seq.finish(SequenceStatus::FinishedPreempted);
        self.block_manager.free_many(&seq.block_table);
        out.preempted.push(seq.id);
        out.finished.push(seq.id);
        self.stats.total_preempts += 1;
        self.stats.total_finished += 1;
        true
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut Sequence> {
        self.running
            .iter_mut()
            .chain(self.waiting.iter_mut())
            .find(|s| s.id == id)
    }

    pub fn get(&self, id: u64) -> Option<&Sequence> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .find(|s| s.id == id)
    }

    pub fn take_finished(&mut self, id: u64) -> Option<Sequence> {
        if let Some(pos) = self.running.iter().position(|s| s.id == id) {
            let seq = self.running.swap_remove(pos);
            self.block_manager.free_many(&seq.block_table);
            return Some(seq);
        }
        if let Some(pos) = self.waiting.iter().position(|s| s.id == id) {
            return Some(self.waiting.swap_remove(pos));
        }
        None
    }

    pub fn finish_sequence(&mut self, id: u64, status: SequenceStatus) {
        if let Some(pos) = self.running.iter().position(|s| s.id == id) {
            let mut seq = self.running.swap_remove(pos);
            seq.finish(status);
            self.block_manager.free_many(&seq.block_table);
        } else if let Some(pos) = self.waiting.iter().position(|s| s.id == id) {
            let mut seq = self.waiting.swap_remove(pos);
            seq.finish(status);
        }
    }

    pub fn abort_request(&mut self, request_id: &str) -> usize {
        let mut n = 0;
        let mut still_running = Vec::new();
        for mut seq in self.running.drain(..) {
            if seq.request_id == request_id && !seq.is_finished() {
                seq.finish(SequenceStatus::FinishedAborted);
                self.block_manager.free_many(&seq.block_table);
                n += 1;
            } else {
                still_running.push(seq);
            }
        }
        self.running = still_running;

        let mut still_waiting = Vec::new();
        for mut seq in self.waiting.drain(..) {
            if seq.request_id == request_id && !seq.is_finished() {
                seq.finish(SequenceStatus::FinishedAborted);
                n += 1;
            } else {
                still_waiting.push(seq);
            }
        }
        self.waiting = still_waiting;
        n
    }
}
