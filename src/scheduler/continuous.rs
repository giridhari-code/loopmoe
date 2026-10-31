//! Continuous batching scheduler
//!
//! Patterns from DeepInfer `Scheduler` (csrc/engine/scheduler.hpp):
//! - waiting / running queues
//! - allocate blocks before admit
//! - prefer running decode, then new prefill
//! - preempt lowest-priority when capacity tight

use super::block_manager::{AllocStatus, BlockManager};
use super::sequence::{Sequence, SequenceStatus};

#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_num_seqs: usize,
    pub max_num_batched_tokens: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
        }
    }
}

#[derive(Debug)]
pub struct SchedulerOutputs {
    pub scheduled: Vec<u64>,
    pub prompt_run: bool,
    pub num_batched_tokens: usize,
    pub finished: Vec<u64>,
}

impl SchedulerOutputs {
    pub fn is_empty(&self) -> bool {
        self.scheduled.is_empty()
    }
}

pub struct ContinuousScheduler {
    cfg: SchedulerConfig,
    block_manager: BlockManager,
    waiting: Vec<Sequence>,
    running: Vec<Sequence>,
    next_id: u64,
}

impl ContinuousScheduler {
    pub fn new(cfg: SchedulerConfig, num_gpu_blocks: usize, block_size: usize) -> Self {
        Self {
            cfg,
            block_manager: BlockManager::new(num_gpu_blocks, block_size),
            waiting: Vec::new(),
            running: Vec::new(),
            next_id: 1,
        }
    }

    pub fn block_manager(&self) -> &BlockManager {
        &self.block_manager
    }

    pub fn block_manager_mut(&mut self) -> &mut BlockManager {
        &mut self.block_manager
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

    /// Main schedule step — returns ids of sequences to run this step.
    pub fn schedule(&mut self) -> SchedulerOutputs {
        let mut scheduled_ids = Vec::new();
        let mut finished_ids = Vec::new();
        let mut num_batched_tokens = 0usize;
        let mut prompt_run = false;

        // 1. Reap finished from running + free their blocks
        let mut still_running = Vec::with_capacity(self.running.len());
        for seq in self.running.drain(..) {
            if seq.is_finished() {
                finished_ids.push(seq.id);
                self.block_manager.free_many(&seq.block_table);
            } else {
                still_running.push(seq);
            }
        }
        self.running = still_running;

        // 2. Prefer continuing running (decode) sequences
        for seq in self.running.iter_mut() {
            if scheduled_ids.len() >= self.cfg.max_num_seqs {
                break;
            }
            let tokens = 1; // decode step
            if num_batched_tokens + tokens > self.cfg.max_num_batched_tokens {
                break;
            }
            let need = self
                .block_manager
                .blocks_for_tokens(seq.len() + 1)
                .saturating_sub(seq.block_table.len());
            if need > 0 {
                match self.block_manager.can_allocate(need) {
                    AllocStatus::Ok => {
                        if let Some(extra) = self.block_manager.allocate(need) {
                            seq.block_table.extend(extra);
                        } else {
                            continue;
                        }
                    }
                    AllocStatus::Later | AllocStatus::Never => continue,
                }
            }
            scheduled_ids.push(seq.id);
            num_batched_tokens += tokens;
        }

        // 3. Admit waiting prefills if budget remains
        let mut admitted = Vec::new();
        let mut waiting_keep = Vec::new();
        for mut seq in self.waiting.drain(..) {
            if scheduled_ids.len() + admitted.len() >= self.cfg.max_num_seqs {
                waiting_keep.push(seq);
                continue;
            }
            let prompt_tokens = seq.prompt_len.max(1);
            if num_batched_tokens + prompt_tokens > self.cfg.max_num_batched_tokens
                && !scheduled_ids.is_empty()
            {
                waiting_keep.push(seq);
                continue;
            }
            let need = self.block_manager.blocks_for_tokens(seq.len().max(1));
            match self.block_manager.can_allocate(need) {
                AllocStatus::Ok => {
                    if let Some(blocks) = self.block_manager.allocate(need) {
                        seq.block_table = blocks;
                        seq.status = SequenceStatus::Running;
                        num_batched_tokens += prompt_tokens;
                        prompt_run = true;
                        admitted.push(seq);
                    } else {
                        waiting_keep.push(seq);
                    }
                }
                AllocStatus::Later | AllocStatus::Never => {
                    waiting_keep.push(seq);
                }
            }
        }
        self.waiting = waiting_keep;
        for seq in admitted {
            scheduled_ids.push(seq.id);
            self.running.push(seq);
        }

        SchedulerOutputs {
            scheduled: scheduled_ids,
            prompt_run,
            num_batched_tokens,
            finished: finished_ids,
        }
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

    /// Mark sequence finished and free its blocks.
    pub fn finish_sequence(&mut self, id: u64, status: SequenceStatus) {
        if let Some(pos) = self.running.iter().position(|s| s.id == id) {
            let mut seq = self.running.remove(pos);
            seq.finish(status);
            self.block_manager.free_many(&seq.block_table);
        } else if let Some(pos) = self.waiting.iter().position(|s| s.id == id) {
            let mut seq = self.waiting.remove(pos);
            seq.finish(status);
        }
    }

    /// Abort by request_id (DeepInfer-style).
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
