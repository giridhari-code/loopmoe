//! BatchEngine — Orbit O3–O5: chunked prefill, Tesla lane, Pulse B>1, prefix cache

use std::collections::HashMap;

use candle_core::{DType, Tensor};
use rand::{rngs::StdRng, SeedableRng};

use crate::cache::{LayerCaches, MlaCaches, RuntimeCaches};
use crate::engine::inference::InferenceEngine;
use crate::orbit::{
    pack::{PackItem, PackRole, PackedBatch},
    pulse::Pulse,
    BlockLedger, CacheKind, LedgerConfig,
};
use crate::sampler::{is_stop, sample_token, SamplingParams};
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
    pub prefill_chunks: u64,
    pub decodes: u64,
    pub tesla_steps: u64,
    pub tokens_generated: u64,
    pub finished: u64,
    pub preempted: u64,
    pub pulse_decode_forwards: u64,
    pub pulse_prefill_forwards: u64,
    pub prefix_hits: u64,
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
            enable_prefix_cache: true,
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
        let prompt_for_prefix = prompt_tokens.clone();
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
            if led.config().enable_prefix_cache {
                let hits = led.allocate_with_prefix(id, &prompt_for_prefix);
                self.metrics.prefix_hits += hits as u64;
            } else {
                led.register_seq(id);
                let _ = led.ensure_tokens(id, prompt_for_prefix.len().max(1));
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

        if let Some(led) = self.ledger.as_mut() {
            for &id in outputs.finished.iter().chain(outputs.preempted.iter()) {
                led.free_seq(id);
            }
        }

        for chunk in &outputs.prefill_chunks {
            if self
                .scheduler
                .get(chunk.seq_id)
                .map(|s| s.is_finished())
                .unwrap_or(true)
            {
                continue;
            }
            self.run_prefill_chunk(chunk.seq_id, chunk.start, chunk.len)?;
            self.metrics.prefill_chunks += 1;
            self.metrics.prefills += 1;
        }

        let mut decode_items = Vec::new();
        for &id in &outputs.decode_ids {
            if self.scheduler.get(id).map(|s| s.is_finished()).unwrap_or(true) {
                continue;
            }
            if self.scheduler.get(id).map(|s| s.is_prompt()).unwrap_or(false) {
                continue;
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

        if !decode_items.is_empty() {
            let pack = PackedBatch {
                num_prefill: 0,
                num_decode: decode_items.len(),
                num_tokens: decode_items.len(),
                items: decode_items,
                slots: Default::default(),
            };
            self.run_pulse_decode(&pack)?;
        }

        for &id in &outputs.tesla_ids {
            if self.scheduler.get(id).map(|s| s.is_finished()).unwrap_or(true) {
                continue;
            }
            if self.scheduler.get(id).map(|s| s.is_prompt()).unwrap_or(false) {
                continue;
            }
            // Tesla only if config present; else fall back plain path next step
            if self.tesla.as_ref().map(|t| t.enabled).unwrap_or(false) {
                self.run_decode_tesla(id)?;
                self.metrics.tesla_steps += 1;
                self.metrics.decodes += 1;
            }
        }

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
            prefill_chunks = self.metrics.prefill_chunks,
            decodes = self.metrics.decodes,
            tesla = self.metrics.tesla_steps,
            prefix_hits = self.metrics.prefix_hits,
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

    fn run_prefill_chunk(&mut self, id: u64, start: usize, len: usize) -> Result<()> {
        if len == 0 {
            return Ok(());
        }
        let (chunk_tokens, prompt_len, sampling, max_tokens, prompt_snapshot) = {
            let seq = self
                .scheduler
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("missing seq {id}"))?;
            let end = (start + len).min(seq.prompt_len).min(seq.tokens.len());
            if start >= end {
                return Ok(());
            }
            (
                seq.tokens[start..end].to_vec(),
                seq.prompt_len,
                seq.sampling.clone(),
                seq.stop.max_tokens,
                seq.tokens[..seq.prompt_len.min(seq.tokens.len())].to_vec(),
            )
        };
        let actual_len = chunk_tokens.len();
        let is_last = start + actual_len >= prompt_len;

        let rt = self
            .runtimes
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("missing runtime {id}"))?;
        let device = &self.engine.device;
        let input = Tensor::new(chunk_tokens.as_slice(), device)?.unsqueeze(0)?;
        let logits = squeeze_logits(&self.engine.model.forward(
            &input,
            start,
            Some(&mut rt.caches),
        )?)?;

        if let Some(seq) = self.scheduler.get_mut(id) {
            seq.mark_computed(actual_len);
        }
        if let Some(led) = self.ledger.as_mut() {
            let _ = led.ensure_tokens(id, start + actual_len);
        }

        if is_last {
            // Seal prompt pages into prefix index for future requests
            if let Some(led) = self.ledger.as_mut() {
                led.seal_seq_prompt(id, &prompt_snapshot);
            }

            let next = sample_token(&logits, &sampling, &rt.generated, &mut rt.rng)?;
            rt.generated.push(next);
            self.metrics.tokens_generated += 1;
            if let Some(led) = self.ledger.as_mut() {
                let _ = led.append_token_slot(id);
            }
            if let Some(seq) = self.scheduler.get_mut(id) {
                seq.append_token(next);
                if is_stop(next, &sampling) || seq.generated_len() >= max_tokens {
                    seq.finish(if seq.generated_len() >= max_tokens {
                        SequenceStatus::FinishedLengthCapped
                    } else {
                        SequenceStatus::FinishedStopped
                    });
                }
            }
        }
        Ok(())
    }

    fn run_pulse_decode(&mut self, pack: &PackedBatch) -> Result<()> {
        let mut caches_map = HashMap::new();
        let mut gen_map = HashMap::new();
        let mut rng_map = HashMap::new();
        let mut mtp_map = HashMap::new();
        let mut samp_map = HashMap::new();

        for item in &pack.items {
            if let Some(rt) = self.runtimes.remove(&item.seq_id) {
                gen_map.insert(item.seq_id, rt.generated);
                rng_map.insert(item.seq_id, rt.rng);
                caches_map.insert(item.seq_id, rt.caches);
                mtp_map.insert(item.seq_id, rt.mtp);
            }
            if let Some(seq) = self.scheduler.get(item.seq_id) {
                samp_map.insert(item.seq_id, seq.sampling.clone());
            }
        }

        let pulse = Pulse::new(&self.engine.model, &self.engine.device);
        let pout = pulse.run(pack, &mut caches_map, &mut gen_map, &mut rng_map, &samp_map)?;
        self.metrics.pulse_decode_forwards += pout.metrics.decode_forwards;

        for (id, new_toks) in &pout.new_tokens {
            for &t in new_toks {
                self.metrics.tokens_generated += 1;
                self.metrics.decodes += 1;
                if let Some(seq) = self.scheduler.get_mut(*id) {
                    seq.append_token(t);
                    seq.mark_computed(1);
                    let max_tokens = seq.stop.max_tokens;
                    let sampling = seq.sampling.clone();
                    if is_stop(t, &sampling) || seq.generated_len() >= max_tokens {
                        seq.finish(if seq.generated_len() >= max_tokens {
                            SequenceStatus::FinishedLengthCapped
                        } else {
                            SequenceStatus::FinishedStopped
                        });
                    }
                }
                if let Some(led) = self.ledger.as_mut() {
                    let _ = led.append_token_slot(*id);
                }
            }
        }

        let ids: Vec<u64> = caches_map.keys().copied().collect();
        for id in ids {
            let caches = caches_map.remove(&id).unwrap();
            let generated = gen_map.remove(&id).unwrap_or_default();
            let rng = rng_map
                .remove(&id)
                .unwrap_or_else(|| StdRng::seed_from_u64(id));
            let mtp = mtp_map.remove(&id).unwrap_or_default();
            self.runtimes.insert(
                id,
                SeqRuntime {
                    caches,
                    mtp,
                    generated,
                    rng,
                },
            );
        }
        Ok(())
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

fn squeeze_logits(logits: &Tensor) -> candle_core::Result<Tensor> {
    use candle_core::IndexOp;
    match logits.dims() {
        [_] => Ok(logits.clone()),
        [1, _] => logits.squeeze(0),
        [1, 1, _] => logits.squeeze(0)?.squeeze(0),
        dims if dims.len() == 2 => logits.i(0),
        _ => logits.flatten_all(),
    }
}
