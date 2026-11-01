//! OrbitCore — schedule + ledger + pulse (O3-aware).

use std::collections::HashMap;

use candle_core::Device;
use rand::rngs::StdRng;

use crate::cache::RuntimeCaches;
use crate::model::LoopedMoE;
use crate::sampler::SamplingParams;
use crate::scheduler::{SchedulerConfig, SchedulerOutputs, Sequence, SequenceStatus};
use crate::Result;

use super::ledger::{BlockLedger, LedgerConfig};
use super::pack::{PackItem, PackRole, PackedBatch};
use super::pulse::{Pulse, PulseMetrics, PulseOutput};
use super::sched::OrbitSched;

pub struct OrbitCore {
    pub sched: OrbitSched,
    device: Device,
}

impl OrbitCore {
    pub fn new(
        sched_cfg: SchedulerConfig,
        num_gpu_blocks: usize,
        block_size: usize,
        ledger_cfg: Option<LedgerConfig>,
        device: Device,
    ) -> Self {
        let ledger = ledger_cfg.map(BlockLedger::new);
        Self {
            sched: OrbitSched::new(sched_cfg, num_gpu_blocks, block_size, ledger),
            device,
        }
    }

    pub fn add_sequence(&mut self, seq: Sequence) -> u64 {
        self.sched.add_sequence(seq)
    }

    pub fn has_unfinished(&self) -> bool {
        self.sched.has_unfinished()
    }

    pub fn build_pack_from_outputs(
        &self,
        out: &SchedulerOutputs,
        resolve: &dyn Fn(u64) -> Option<PackResolve>,
    ) -> PackedBatch {
        let mut items = Vec::new();
        let mut num_prefill = 0;
        let mut num_decode = 0;
        let mut num_tokens = 0;
        let mut ranges = Vec::new();

        for chunk in &out.prefill_chunks {
            if let Some(PackResolve::Prefill { tokens, position }) = resolve(chunk.seq_id) {
                // Prefer scheduler chunk bounds when resolve gives full prompt
                let start = chunk.start;
                let end = (chunk.start + chunk.len).min(tokens.len());
                let slice = if start < end {
                    tokens[start..end].to_vec()
                } else {
                    tokens
                };
                let len = slice.len();
                num_tokens += len;
                num_prefill += 1;
                ranges.push((chunk.seq_id, position.max(start), len));
                items.push(PackItem {
                    seq_id: chunk.seq_id,
                    role: PackRole::Prefill,
                    token_ids: slice,
                    position: position.max(start),
                });
            }
        }

        for &id in &out.decode_ids {
            if let Some(PackResolve::Decode {
                last_token,
                position,
            }) = resolve(id)
            {
                num_tokens += 1;
                num_decode += 1;
                ranges.push((id, position, 1));
                items.push(PackItem {
                    seq_id: id,
                    role: PackRole::Decode,
                    token_ids: vec![last_token],
                    position,
                });
            }
        }
        // Tesla ids are intentionally not packed into Pulse (special path).

        let slots = self
            .sched
            .ledger()
            .map(|led| led.build_slot_mapping(&ranges))
            .unwrap_or_default();

        PackedBatch {
            items,
            slots,
            num_decode,
            num_prefill,
            num_tokens,
        }
    }

    /// Schedule → pack (prefill chunks + plain decode) → Pulse.
    /// Returns outputs including `tesla_ids` for the caller to run Tesla.
    pub fn step(
        &mut self,
        model: &LoopedMoE,
        caches: &mut HashMap<u64, RuntimeCaches>,
        generated: &mut HashMap<u64, Vec<u32>>,
        rngs: &mut HashMap<u64, StdRng>,
        sampling: &HashMap<u64, SamplingParams>,
        resolve: &dyn Fn(u64) -> Option<PackResolve>,
    ) -> Result<(SchedulerOutputs, PulseOutput)> {
        let out = self.sched.schedule();
        let pack = self.build_pack_from_outputs(&out, resolve);
        let pulse = Pulse::new(model, &self.device);
        let pout = if pack.is_empty() {
            PulseOutput {
                new_tokens: HashMap::new(),
                metrics: PulseMetrics::default(),
            }
        } else {
            pulse.run(&pack, caches, generated, rngs, sampling)?
        };
        Ok((out, pout))
    }

    pub fn finish(&mut self, id: u64, status: SequenceStatus) {
        self.sched.finish_sequence(id, status);
    }

    pub fn seal_prefix(&mut self, id: u64, prompt: &[u32]) {
        self.sched.seal_prefix(id, prompt);
    }
}

#[derive(Debug, Clone)]
pub enum PackResolve {
    Prefill { tokens: Vec<u32>, position: usize },
    Decode { last_token: u32, position: usize },
}

#[derive(Debug, Default, Clone)]
pub struct OrbitMetrics {
    pub schedule_calls: u64,
    pub pulse: PulseMetrics,
}
