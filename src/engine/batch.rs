//! BatchEngine — continuous scheduler + Orbit Pulse (O2) step loop.

use std::collections::HashMap;

use candle_core::DType;
use rand::{rngs::StdRng, SeedableRng};

use crate::cache::{LayerCaches, MlaCaches, RuntimeCaches};
use crate::engine::inference::InferenceEngine;
use crate::orbit::{
    pack::{PackItem, PackRole, PackedBatch},
    pulse::Pulse,
    BlockLedger, LedgerConfig, CacheKind,
};
use crate::sampler::{is_stop, SamplingParams};
use crate::scheduler::{
    ContinuousScheduler, SchedulerConfig, Sequence, SequenceStatus,
};
use crate::tesla::{draft_tokens, verify_and_accept, MtpState, TeslaConfig};
use crate::Result;

struct SeqRuntime {
    caches: RuntimeCaches,
    mtp: MtpState,
    generated: Vec<u32>,
    rng: StdRng,
}

#[derive(Debug, Default, Clone)]
pub struct BatchMetrics {
    pub steps: u64,
    pub prefills: u64,
    pub decodes: u64,
    pub tesla_steps: u64,
    pub tokens_generated: u64,
    pub finished: u64,
    pub preempted: u64,
    pub pulse_decode_forwards: u64,
    pub pulse_prefill_forwards: u64,
}

pub struct BatchEngine {
    engine: InferenceEngine,
    scheduler: ContinuousScheduler,
    ledger: Option<BlockLedger>,
    tesla: Option<TeslaConfig>,
    runtimes: HashMap<u64, SeqRuntime>,
    pub results: HashMap<u64, Vec<u32>>,
    pub metrics: BatchMetrics,
    dtype: DType,
}

impl BatchEngine {
    pub fn new(
        engine: InferenceEngine,
        sched_cfg: SchedulerConfig,
        num_gpu_blocks: usize,
        block_size: usize,
        tesla: Option<TeslaConfig>,
    ) -> Self {
        let ledger_cfg = LedgerConfig {
            num_pages: num_gpu_blocks.max(1),
            block_size: block_size.max(1),
            kind: if engine.config.use_mla {
                CacheKind::Mla
            } else {
                CacheKind::Gqa
            },
            num_layers: engine.config.num_layers,
            num_kv_heads: engine.config.num_key_value_heads,
            head_dim: engine.config.head_dim,
            kv_lora_rank: engine.config.kv_lora_rank,
            rope_dim: engine.config.qk_rope_head_dim,
        };
        Self {
            engine,
            scheduler: ContinuousScheduler::new(sched_cfg, num_gpu_blocks, block_size),
            ledger: Some(BlockLedger::new(ledger_cfg)),
            tesla,
            runtimes: HashMap::new(),
            results: HashMap::new(),
            metrics: BatchMetrics::default(),
            dtype: DType::BF16,
        }
    }

    pub fn scheduler(&self) -> &ContinuousScheduler {
        &self.scheduler
    }

    pub fn scheduler_mut(&mut self) -> &mut ContinuousScheduler {
        &mut self.scheduler
    }

    pub fn ledger(&self) -> Option<&BlockLedger> {
        self.ledger.as_ref()
    }

    pub fn engine(&self) -> &InferenceEngine {
        &self.engine
    }

    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<u32>,
        sampling: SamplingParams,
        max_tokens: usize,
        request_id: Option<String>,
    ) -> u64 {
        let mut seq = Sequence::new(0, prompt_tokens);
        if let Some(rid) = request_id {
            seq = seq.with_request_id(rid);
        }
        seq.sampling = sampling.clone();
        seq.stop.max_tokens = max_tokens;
        if let Some(ref t) = self.tesla {
            if t.enabled {
                seq.draft_len = Some(t.effective_nextn());
            }
        }
        let id = self.scheduler.add_sequence(seq);

        if let Some(led) = self.ledger.as_mut() {
            led.register_seq(id);
            if let Some(s) = self.scheduler.get(id) {
                let _ = led.ensure_tokens(id, s.prompt_len.max(1));
            }
        }

        let capacity = {
            let s = self.scheduler.get(id).unwrap();
            (s.prompt_len + max_tokens)
                .min(self.engine.config.max_seq_len)
                .max(s.prompt_len + 1)
        };

        let caches = if self.engine.config.use_mla {
            RuntimeCaches::Mla(MlaCaches::with_capacity(
                self.engine.config.num_layers,
                1,
                self.engine.config.kv_lora_rank,
                self.engine.config.qk_rope_head_dim,
                capacity,
                self.dtype,
                &self.engine.device,
            ))
        } else {
            RuntimeCaches::Gqa(LayerCaches::with_capacity(
                self.engine.config.num_layers,
                1,
                self.engine.config.num_key_value_heads,
                self.engine.config.head_dim,
                capacity,
                self.dtype,
                &self.engine.device,
            ))
        };

        let seed = sampling.seed.unwrap_or(id);
        self.runtimes.insert(
            id,
            SeqRuntime {
                caches,
                mtp: MtpState::default(),
                generated: Vec::with_capacity(max_tokens),
                rng: StdRng::seed_from_u64(seed),
            },
        );
        id
    }

    pub fn has_unfinished(&self) -> bool {
        self.scheduler.has_unfinished()
    }

    pub fn step(&mut self) -> Result<Vec<u64>> {
        self.metrics.steps += 1;
        let outputs = self.scheduler.schedule();
        self.metrics.preempted += outputs.preempted.len() as u64;

        // Ledger sync
        if let Some(led) = self.ledger.as_mut() {
            for &id in outputs.finished.iter().chain(outputs.preempted.iter()) {
                led.free_seq(id);
            }
            for &id in outputs.decode_ids.iter().chain(outputs.prefill_ids.iter()) {
                if let Some(s) = self.scheduler.get(id) {
                    let need = if outputs.prefill_ids.contains(&id) {
                        s.prompt_len.max(1)
                    } else {
                        s.len() + s.draft_len.unwrap_or(1)
                    };
                    let _ = led.ensure_tokens(id, need);
                }
            }
        }

        let use_tesla = self.tesla.as_ref().map(|t| t.enabled).unwrap_or(false);

        // Build PackedBatch for non-Tesla paths
        let mut prefill_items = Vec::new();
        let mut decode_items = Vec::new();
        for &id in &outputs.prefill_ids {
            if self.scheduler.get(id).map(|s| s.is_finished()).unwrap_or(true) {
                continue;
            }
            if let Some(seq) = self.scheduler.get(id) {
                prefill_items.push(PackItem {
                    seq_id: id,
                    role: PackRole::Prefill,
                    token_ids: seq.tokens[..seq.prompt_len].to_vec(),
                    position: 0,
                });
            }
        }
        for &id in &outputs.decode_ids {
            if self.scheduler.get(id).map(|s| s.is_finished()).unwrap_or(true) {
                continue;
            }
            if self.scheduler.get(id).map(|s| s.is_prompt()).unwrap_or(false) {
                if let Some(seq) = self.scheduler.get(id) {
                    prefill_items.push(PackItem {
                        seq_id: id,
                        role: PackRole::Prefill,
                        token_ids: seq.tokens[..seq.prompt_len].to_vec(),
                        position: 0,
                    });
                }
                continue;
            }
            if use_tesla {
                continue; // Tesla handled separately
            }
            if let Some(seq) = self.scheduler.get(id) {
                let last = *seq.tokens.last().unwrap_or(&0);
                let pos = self
                    .runtimes
                    .get(&id)
                    .map(|r| r.caches.current_len())
                    .unwrap_or(0);
                decode_items.push(PackItem {
                    seq_id: id,
                    role: PackRole::Decode,
                    token_ids: vec![last],
                    position: pos,
                });
            }
        }

        let slots = if let Some(led) = self.ledger.as_ref() {
            let mut ranges = Vec::new();
            for it in prefill_items.iter().chain(decode_items.iter()) {
                ranges.push((it.seq_id, it.position, it.token_ids.len()));
            }
            led.build_slot_mapping(&ranges)
        } else {
            Default::default()
        };

        let pack = PackedBatch {
            num_prefill: prefill_items.len(),
            num_decode: decode_items.len(),
            num_tokens: prefill_items
                .iter()
                .chain(decode_items.iter())
                .map(|i| i.token_ids.len())
                .sum(),
            items: {
                let mut v = prefill_items;
                v.extend(decode_items);
                v
            },
            slots,
        };

        // Pulse for plain prefill + decode
        if !pack.is_empty() {
            let mut caches_map = HashMap::new();
            let mut gen_map = HashMap::new();
            let mut rng_map = HashMap::new();
            let mut samp_map = HashMap::new();

            for item in &pack.items {
                if let Some(rt) = self.runtimes.remove(&item.seq_id) {
                    gen_map.insert(item.seq_id, rt.generated);
                    rng_map.insert(item.seq_id, rt.rng);
                    caches_map.insert(item.seq_id, rt.caches);
                }
                if let Some(seq) = self.scheduler.get(item.seq_id) {
                    samp_map.insert(item.seq_id, seq.sampling.clone());
                }
            }

            let pulse = Pulse::new(&self.engine.model, &self.engine.device);
            let pout = pulse.run(&pack, &mut caches_map, &mut gen_map, &mut rng_map, &samp_map)?;
            self.metrics.pulse_decode_forwards += pout.metrics.decode_forwards;
            self.metrics.pulse_prefill_forwards += pout.metrics.prefill_forwards;

            for (id, new_toks) in pout.new_tokens {
                for &t in &new_toks {
                    self.metrics.tokens_generated += 1;
                    if let Some(seq) = self.scheduler.get_mut(id) {
                        let is_prefill = pack
                            .items
                            .iter()
                            .any(|i| i.seq_id == id && i.role == PackRole::Prefill);
                        if is_prefill {
                            seq.append_token(t);
                            seq.mark_computed(seq.prompt_len);
                            self.metrics.prefills += 1;
                        } else {
                            seq.append_token(t);
                            seq.mark_computed(1);
                            self.metrics.decodes += 1;
                        }
                        let max_tokens = seq.stop.max_tokens;
                        let sampling = seq.sampling.clone();
                        if is_stop(t, &sampling) || seq.generated_len() >= max_tokens {
                            let status = if seq.generated_len() >= max_tokens {
                                SequenceStatus::FinishedLengthCapped
                            } else {
                                SequenceStatus::FinishedStopped
                            };
                            seq.finish(status);
                        }
                    }
                    if let Some(led) = self.ledger.as_mut() {
                        let _ = led.append_token_slot(id);
                    }
                }
                // restore runtime
                if let (Some(caches), Some(generated), Some(rng)) = (
                    caches_map.remove(&id),
                    gen_map.remove(&id),
                    rng_map.remove(&id),
                ) {
                    self.runtimes.insert(
                        id,
                        SeqRuntime {
                            caches,
                            mtp: self
                                .runtimes
                                .remove(&id)
                                .map(|r| r.mtp)
                                .unwrap_or_default(),
                            generated,
                            rng,
                        },
                    );
                }
            }
            // restore any left
            for (id, caches) in caches_map {
                let generated = gen_map.remove(&id).unwrap_or_default();
                let rng = rng_map
                    .remove(&id)
                    .unwrap_or_else(|| StdRng::seed_from_u64(id));
                self.runtimes.insert(
                    id,
                    SeqRuntime {
                        caches,
                        mtp: MtpState::default(),
                        generated,
                        rng,
                    },
                );
            }
        }

        // Tesla decode path (still per-seq)
        if use_tesla {
            for &id in &outputs.decode_ids {
                if self.scheduler.get(id).map(|s| s.is_finished()).unwrap_or(true) {
                    continue;
                }
                if self.scheduler.get(id).map(|s| s.is_prompt()).unwrap_or(false) {
                    continue;
                }
                self.run_decode_tesla(id)?;
                self.metrics.tesla_steps += 1;
                self.metrics.decodes += 1;
            }
        }

        // Finish reap
        let newly_done: Vec<u64> = self
            .scheduler
            .running_ids()
            .into_iter()
            .filter(|&id| {
                self.scheduler
                    .get(id)
                    .map(|s| s.is_finished())
                    .unwrap_or(false)
            })
            .collect();
        for id in newly_done {
            if let Some(seq) = self.scheduler.get(id) {
                let status = seq.status;
                self.scheduler.finish_sequence(id, status);
            }
        }

        let mut done = Vec::new();
        for seq in self.scheduler.drain_finished() {
            let id = seq.id;
            if let Some(led) = self.ledger.as_mut() {
                led.free_seq(id);
            }
            self.results.insert(id, seq.tokens);
            self.runtimes.remove(&id);
            self.metrics.finished += 1;
            done.push(id);
        }
        Ok(done)
    }

    pub fn run_all(&mut self) -> Result<HashMap<u64, Vec<u32>>> {
        while self.has_unfinished() {
            let _ = self.step()?;
        }
        for seq in self.scheduler.drain_finished() {
            self.results.insert(seq.id, seq.tokens);
            self.runtimes.remove(&seq.id);
        }
        tracing::info!(
            steps = self.metrics.steps,
            prefills = self.metrics.prefills,
            decodes = self.metrics.decodes,
            pulse_dec = self.metrics.pulse_decode_forwards,
            pulse_pref = self.metrics.pulse_prefill_forwards,
            tesla = self.metrics.tesla_steps,
            finished = self.metrics.finished,
            tokens = self.metrics.tokens_generated,
            "BatchEngine done"
        );
        Ok(std::mem::take(&mut self.results))
    }

    pub fn tokens(&self, id: u64) -> Option<Vec<u32>> {
        if let Some(t) = self.results.get(&id) {
            return Some(t.clone());
        }
        self.scheduler.get(id).map(|s| s.tokens.clone())
    }

    pub fn mtp_stats(&self, id: u64) -> Option<&MtpState> {
        self.runtimes.get(&id).map(|r| &r.mtp)
    }

    fn run_decode_tesla(&mut self, id: u64) -> Result<()> {
        let cfg = self
            .tesla
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("tesla config missing"))?
            .clone();

        let (sampling, max_tokens, last_token) = {
            let seq = self
                .scheduler
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("missing seq {id}"))?;
            let last = *seq.tokens.last().unwrap_or(&0);
            (seq.sampling.clone(), seq.stop.max_tokens, last)
        };

        let device = self.engine.device.clone();
        let rt = self
            .runtimes
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("missing runtime {id}"))?;

        let pos = rt.caches.current_len();
        let lens_ckpt = rt.caches.checkpoint_lens();

        let draft = draft_tokens(
            &self.engine.model,
            &mut rt.caches,
            last_token,
            pos,
            &device,
            &cfg,
            &sampling,
            &rt.generated,
            &mut rt.rng,
        )?;
        rt.mtp.set_draft(
            draft.tokens.clone(),
            draft.confidences.clone(),
            draft.margins.clone(),
        );

        let vlen = rt.mtp.schedule_verify_len(&cfg);
        let (accepted, stats) = verify_and_accept(
            &self.engine.model,
            &mut rt.caches,
            &lens_ckpt,
            last_token,
            pos,
            &draft.tokens,
            vlen,
            &device,
            &sampling,
            &rt.generated,
            &mut rt.rng,
        )?;

        rt.mtp
            .on_verify_result(stats.accepted, stats.scheduled, &cfg);

        for &t in &accepted {
            rt.generated.push(t);
            self.metrics.tokens_generated += 1;
            if let Some(led) = self.ledger.as_mut() {
                let _ = led.append_token_slot(id);
            }
            if let Some(seq) = self.scheduler.get_mut(id) {
                seq.append_token(t);
                seq.mark_computed(1);
                if is_stop(t, &sampling) || seq.generated_len() >= max_tokens {
                    seq.finish(if seq.generated_len() >= max_tokens {
                        SequenceStatus::FinishedLengthCapped
                    } else {
                        SequenceStatus::FinishedStopped
                    });
                    break;
                }
            }
        }
        Ok(())
    }
}
