//! Continuous batching scheduler — O3: chunked prefill + Tesla lane

use super::block_manager::{AllocStatus, BlockManager};
use super::sequence::{Sequence, SequenceStatus};

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
    pub shortest_first: bool,
    pub enable_preempt: bool,
    /// Max prompt tokens processed per seq per step (chunked prefill).
    pub prefill_chunk_size: usize,
    /// Default Tesla draft estimate when seq.draft_len is set but 0.
    pub tesla_draft_estimate: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            shortest_first: true,
            enable_preempt: true,
            prefill_chunk_size: 512,
            tesla_draft_estimate: 5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefillChunk {
    pub seq_id: u64,
    /// Index into prompt tokens.
    pub start: usize,
    pub len: usize,
}

#[derive(Debug, Default)]
pub struct SchedulerOutputs {
    /// Chunked prefill work this step.
    pub prefill_chunks: Vec<PrefillChunk>,
    /// Plain decode (1 token budget each).
    pub decode_ids: Vec<u64>,
    /// Tesla speculative decode (budget = draft_len).
    pub tesla_ids: Vec<u64>,
    pub num_batched_tokens: usize,
    pub finished: Vec<u64>,
    pub preempted: Vec<u64>,
}

impl SchedulerOutputs {
    pub fn is_empty(&self) -> bool {
        self.prefill_chunks.is_empty() && self.decode_ids.is_empty() && self.tesla_ids.is_empty()
    }

    pub fn num_scheduled(&self) -> usize {
        self.prefill_chunks.len() + self.decode_ids.len() + self.tesla_ids.len()
    }

    /// Back-compat: all prefill seq ids.
    pub fn prefill_ids(&self) -> Vec<u64> {
        self.prefill_chunks.iter().map(|c| c.seq_id).collect()
    }
}

pub struct ContinuousScheduler {
    cfg: SchedulerConfig,
    block_manager: BlockManager,
    waiting: Vec<Sequence>,
    running: Vec<Sequence>,
    finished_buf: Vec<Sequence>,
    next_id: u64,
    pub stats: SchedulerStats,
}

#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    pub schedule_calls: u64,
    pub total_prefills_admitted: u64,
    pub total_prefill_chunks: u64,
    pub total_decodes_scheduled: u64,
    pub total_tesla_scheduled: u64,
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
            finished_buf: Vec::new(),
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

    #[inline]
    fn chunk_size(&self) -> usize {
        self.cfg.prefill_chunk_size.max(1)
    }

    /// Tokens remaining in prompt not yet computed.
    #[inline]
    fn prefill_remaining(seq: &Sequence) -> usize {
        seq.prompt_len.saturating_sub(seq.num_computed_tokens)
    }

    #[inline]
    fn prefill_chunk_for(seq: &Sequence, chunk_size: usize) -> PrefillChunk {
        let start = seq.num_computed_tokens.min(seq.prompt_len);
        let rem = seq.prompt_len.saturating_sub(start);
        let len = rem.min(chunk_size).max(if rem > 0 { 1 } else { 0 });
        PrefillChunk {
            seq_id: seq.id,
            start,
            len,
        }
    }

    #[inline]
    fn is_tesla(seq: &Sequence) -> bool {
        seq.draft_len.map(|d| d > 0).unwrap_or(false)
    }

    #[inline]
    fn decode_token_estimate(&self, seq: &Sequence) -> usize {
        if Self::is_tesla(seq) {
            seq.draft_len
                .unwrap_or(self.cfg.tesla_draft_estimate)
                .max(1)
        } else {
            1
        }
    }

    pub fn schedule(&mut self) -> SchedulerOutputs {
        self.stats.schedule_calls += 1;
        let mut out = SchedulerOutputs::default();
        let chunk_size = self.chunk_size();

        // ── 1. Reap finished ──────────────────────────────────────────────
        let mut still = Vec::with_capacity(self.running.len());
        for seq in self.running.drain(..) {
            if seq.is_finished() {
                out.finished.push(seq.id);
                self.block_manager.free_many(&seq.block_table);
                self.finished_buf.push(seq);
                self.stats.total_finished += 1;
            } else {
                still.push(seq);
            }
        }
        self.running = still;

        let max_seqs = self.cfg.max_num_seqs;
        let max_tok = self.cfg.max_num_batched_tokens;
        let mut num_tokens = 0usize;
        let mut scheduled_ids: Vec<u64> = Vec::new();

        // ── 2. Decode + Tesla (running, prompt done) ─────────────────────
        let mut candidates: Vec<(u64, usize, usize, bool)> = Vec::new();
        // (id, est, extra_blocks, is_tesla)
        for seq in &self.running {
            if seq.is_prompt() {
                continue;
            }
            let est = self.decode_token_estimate(seq);
            let need = self
                .block_manager
                .blocks_for_tokens(seq.len() + est)
                .saturating_sub(seq.block_table.len());
            candidates.push((seq.id, est, need, Self::is_tesla(seq)));
        }

        for (id, est, need, is_tesla) in candidates {
            if scheduled_ids.len() >= max_seqs {
                break;
            }
            if num_tokens + est > max_tok && !scheduled_ids.is_empty() {
                break;
            }
            if need > 0 && !self.try_grow_blocks(id, need, &mut out) {
                continue;
            }
            scheduled_ids.push(id);
            num_tokens += est;
            if is_tesla {
                out.tesla_ids.push(id);
                self.stats.total_tesla_scheduled += 1;
            } else {
                out.decode_ids.push(id);
                self.stats.total_decodes_scheduled += 1;
            }
        }

        // ── 3. Continuing chunked prefills (already running) ─────────────
        let mut prefill_candidates: Vec<u64> = self
            .running
            .iter()
            .filter(|s| s.is_prompt() && !s.is_finished())
            .map(|s| s.id)
            .collect();

        for id in prefill_candidates.drain(..) {
            if scheduled_ids.len() >= max_seqs {
                break;
            }
            let Some(seq) = self.running.iter().find(|s| s.id == id) else {
                continue;
            };
            let chunk = Self::prefill_chunk_for(seq, chunk_size);
            if chunk.len == 0 {
                continue;
            }
            if num_tokens + chunk.len > max_tok && !scheduled_ids.is_empty() {
                break;
            }
            // Ensure blocks for prompt progress
            let need_tokens = chunk.start + chunk.len;
            let need = self
                .block_manager
                .blocks_for_tokens(need_tokens)
                .saturating_sub(seq.block_table.len());
            if need > 0 && !self.try_grow_blocks(id, need, &mut out) {
                continue;
            }
            scheduled_ids.push(id);
            num_tokens += chunk.len;
            out.prefill_chunks.push(chunk);
            self.stats.total_prefill_chunks += 1;
        }

        // ── 4. Admit new waiting (first chunk only in budget) ─────────────
        if self.cfg.shortest_first && self.waiting.len() > 1 {
            self.waiting.sort_by_key(|s| s.prompt_len);
        }

        let mut admitted = Vec::new();
        let mut waiting_keep = Vec::with_capacity(self.waiting.len());

        for mut seq in self.waiting.drain(..) {
            let slots = self.running.len() + admitted.len() + scheduled_ids.len();
            if slots >= max_seqs {
                waiting_keep.push(seq);
                continue;
            }
            let first = Self::prefill_chunk_for(&seq, chunk_size);
            let cost = first.len.max(1);
            if num_tokens + cost > max_tok && !scheduled_ids.is_empty() {
                waiting_keep.push(seq);
                continue;
            }

            // Allocate blocks for full prompt up front (stable block table)
            let need = self.block_manager.blocks_for_tokens(seq.prompt_len.max(1));
            match self.block_manager.can_allocate(need) {
                AllocStatus::Ok => {
                    if let Some(blocks) = self.block_manager.allocate(need) {
                        seq.block_table = blocks;
                        seq.status = SequenceStatus::Running;
                        num_tokens += cost;
                        scheduled_ids.push(seq.id);
                        out.prefill_chunks.push(PrefillChunk {
                            seq_id: seq.id,
                            start: first.start,
                            len: first.len,
                        });
                        admitted.push(seq);
                        self.stats.total_prefills_admitted += 1;
                        self.stats.total_prefill_chunks += 1;
                    } else {
                        waiting_keep.push(seq);
                    }
                }
                AllocStatus::Later if self.cfg.enable_preempt => {
                    if self.preempt_one(&mut out) {
                        if let Some(blocks) = self.block_manager.allocate(need) {
                            seq.block_table = blocks;
                            seq.status = SequenceStatus::Running;
                            num_tokens += cost;
                            scheduled_ids.push(seq.id);
                            out.prefill_chunks.push(PrefillChunk {
                                seq_id: seq.id,
                                start: first.start,
                                len: first.len,
                            });
                            admitted.push(seq);
                            self.stats.total_prefills_admitted += 1;
                            self.stats.total_prefill_chunks += 1;
                        } else {
                            waiting_keep.push(seq);
                        }
                    } else {
                        waiting_keep.push(seq);
                    }
                }
                AllocStatus::Never => {
                    seq.finish(SequenceStatus::FinishedIgnored);
                    out.finished.push(seq.id);
                    self.finished_buf.push(seq);
                    self.stats.total_finished += 1;
                }
                _ => waiting_keep.push(seq),
            }
        }
        self.waiting = waiting_keep;
        self.running.extend(admitted);

        out.num_batched_tokens = num_tokens;
        out
    }

    fn try_grow_blocks(&mut self, id: u64, need: usize, out: &mut SchedulerOutputs) -> bool {
        match self.block_manager.can_allocate(need) {
            AllocStatus::Ok => {
                if let Some(extra) = self.block_manager.allocate(need) {
                    if let Some(seq) = self.running.iter_mut().find(|s| s.id == id) {
                        seq.block_table.extend(extra);
                    }
                    true
                } else {
                    false
                }
            }
            AllocStatus::Later if self.cfg.enable_preempt => {
                if !self.preempt_one_except(id, out) {
                    return false;
                }
                if let Some(extra) = self.block_manager.allocate(need) {
                    if let Some(seq) = self.running.iter_mut().find(|s| s.id == id) {
                        seq.block_table.extend(extra);
                    }
                    true
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn preempt_one(&mut self, out: &mut SchedulerOutputs) -> bool {
        self.preempt_one_except(u64::MAX, out)
    }

    fn preempt_one_except(&mut self, except_id: u64, out: &mut SchedulerOutputs) -> bool {
        let mut best: Option<(usize, usize)> = None;
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
        out.decode_ids.retain(|&id| id != seq.id);
        out.tesla_ids.retain(|&id| id != seq.id);
        out.prefill_chunks.retain(|c| c.seq_id != seq.id);
        seq.finish(SequenceStatus::FinishedPreempted);
        self.block_manager.free_many(&seq.block_table);
        out.preempted.push(seq.id);
        out.finished.push(seq.id);
        self.finished_buf.push(seq);
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

    pub fn drain_finished(&mut self) -> Vec<Sequence> {
        std::mem::take(&mut self.finished_buf)
    }

    pub fn take_finished(&mut self, id: u64) -> Option<Sequence> {
        if let Some(pos) = self.finished_buf.iter().position(|s| s.id == id) {
            return Some(self.finished_buf.swap_remove(pos));
        }
        if let Some(pos) = self.running.iter().position(|s| s.id == id) {
            let seq = self.running.swap_remove(pos);
            if seq.is_finished() {
                self.block_manager.free_many(&seq.block_table);
            }
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
            self.finished_buf.push(seq);
        } else if let Some(pos) = self.waiting.iter().position(|s| s.id == id) {
            let mut seq = self.waiting.swap_remove(pos);
            seq.finish(status);
            self.finished_buf.push(seq);
        }
    }

    pub fn abort_request(&mut self, request_id: &str) -> usize {
        let mut n = 0;
        let mut still_running = Vec::new();
        for mut seq in self.running.drain(..) {
            if seq.request_id == request_id && !seq.is_finished() {
                seq.finish(SequenceStatus::FinishedAborted);
                self.block_manager.free_many(&seq.block_table);
                self.finished_buf.push(seq);
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
                self.finished_buf.push(seq);
                n += 1;
            } else {
                still_waiting.push(seq);
            }
        }
        self.waiting = still_waiting;
        n
    }
}
