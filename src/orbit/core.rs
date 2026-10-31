//! OrbitCore — schedule + ledger + pulse in one step API.

use std::collections::HashMap;

use candle_core::Device;
use rand::rngs::StdRng;

use crate::cache::RuntimeCaches;
use crate::model::LoopedMoE;
use crate::sampler::SamplingParams;
use crate::scheduler::{SchedulerConfig, SchedulerOutputs, Sequence, SequenceStatus};
use crate::Result;

use super::ledger::{BlockLedger, LedgerConfig};
use super::pack::PackedBatch;
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

    /// Build PackedBatch from the latest schedule (caller supplies token views).
    pub fn build_pack(
        &self,
        out: &SchedulerOutputs,
        prefill: &[(u64, Vec<u32>, usize)],
        decode: &[(u64, u32, usize)],
    ) -> PackedBatch {
        let slots = if let Some(led) = self.sched.ledger() {
            let mut ranges = Vec::new();
            for &(id, ref toks, pos) in prefill {
                ranges.push((id, pos, toks.len()));
            }
            for &(id, _, pos) in decode {
                ranges.push((id, pos, 1));
            }
            led.build_slot_mapping(&ranges)
        } else {
            Default::default()
        };
        PackedBatch::from_schedule(prefill, decode, slots)
    }

    /// One Orbit step: schedule → pack → pulse.
    pub fn step(
        &mut self,
        model: &LoopedMoE,
        caches: &mut HashMap<u64, RuntimeCaches>,
        generated: &mut HashMap<u64, Vec<u32>>,
        rngs: &mut HashMap<u64, StdRng>,
        sampling: &HashMap<u64, SamplingParams>,
        // resolve prompt tokens / last token / position from caller's seq state
        resolve: &dyn Fn(u64) -> Option<(PackResolve, SamplingParams)>,
    ) -> Result<(SchedulerOutputs, PulseOutput)> {
        let out = self.sched.schedule();

        let mut prefill = Vec::new();
        let mut decode = Vec::new();
        for &id in &out.prefill_ids {
            if let Some((PackResolve::Prefill { tokens, position }, _)) = resolve(id) {
                prefill.push((id, tokens, position));
            }
        }
        for &id in &out.decode_ids {
            if let Some((PackResolve::Decode { last_token, position }, _)) = resolve(id) {
                decode.push((id, last_token, position));
            }
        }

        let pack = self.build_pack(&out, &prefill, &decode);
        let pulse = Pulse::new(model, &self.device);
        let pout = pulse.run(&pack, caches, generated, rngs, sampling)?;
        Ok((out, pout))
    }

    pub fn finish(&mut self, id: u64, status: SequenceStatus) {
        self.sched.finish_sequence(id, status);
    }
}

#[derive(Debug, Clone)]
pub enum PackResolve {
    Prefill { tokens: Vec<u32>, position: usize },
    Decode { last_token: u32, position: usize },
}

/// Aggregate metrics helper.
#[derive(Debug, Default, Clone)]
pub struct OrbitMetrics {
    pub schedule_calls: u64,
    pub pulse: PulseMetrics,
}
