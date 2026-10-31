//! BatchEngine — continuous scheduler + InferenceEngine step loop (optimized).
//!
//! Flow:
//! 1. `add_request` → waiting queue
//! 2. `schedule()` → prefill_ids + decode_ids
//! 3. run prefills, then decodes (plain or Tesla MTP)
//! 4. reap finished → `results`

use std::collections::HashMap;

use candle_core::{Tensor, DType};
use rand::{rngs::StdRng, SeedableRng};

use crate::cache::{LayerCaches, MlaCaches, RuntimeCaches};
use crate::engine::inference::InferenceEngine;
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
    pub decodes: u64,
    pub tesla_steps: u64,
    pub tokens_generated: u64,
    pub finished: u64,
    pub preempted: u64,
}

pub struct BatchEngine {
    engine: InferenceEngine,
    scheduler: ContinuousScheduler,
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
        Self {
            engine,
            scheduler: ContinuousScheduler::new(sched_cfg, num_gpu_blocks, block_size),
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

    /// One optimized schedule + execute step.
    pub fn step(&mut self) -> Result<Vec<u64>> {
        self.metrics.steps += 1;
        let outputs = self.scheduler.schedule();

        // Track preempts from scheduler
        self.metrics.preempted += outputs.preempted.len() as u64;

        // Snapshot prefill / decode lists (owned)
        let prefill_ids = outputs.prefill_ids;
        let decode_ids = outputs.decode_ids;
        let mut finished_ids = outputs.finished;

        // ── Prefills first ────────────────────────────────────────────────
        for &id in &prefill_ids {
            if self.scheduler.get(id).map(|s| s.is_finished()).unwrap_or(true) {
                continue;
            }
            self.run_prefill(id)?;
            self.metrics.prefills += 1;
        }

        // ── Decodes ──────────────────────────────────────────────────────
        let use_tesla = self.tesla.as_ref().map(|t| t.enabled).unwrap_or(false);
        for &id in &decode_ids {
            if self.scheduler.get(id).map(|s| s.is_finished()).unwrap_or(true) {
                continue;
            }
            // Skip if still needs prefill (shouldn't happen with split schedule)
            if self.scheduler.get(id).map(|s| s.is_prompt()).unwrap_or(false) {
                self.run_prefill(id)?;
                self.metrics.prefills += 1;
                continue;
            }
            if use_tesla {
                self.run_decode_tesla(id)?;
                self.metrics.tesla_steps += 1;
            } else {
                self.run_decode_plain(id)?;
            }
            self.metrics.decodes += 1;
        }

        // ── Reap anything finished this step ──────────────────────────────
        for id in self.scheduler.running_ids() {
            if self.scheduler.get(id).map(|s| s.is_finished()).unwrap_or(false) {
                finished_ids.push(id);
            }
        }
        // Also preempted already in finished_ids
        finished_ids.sort_unstable();
        finished_ids.dedup();

        let mut done = Vec::with_capacity(finished_ids.len());
        for id in finished_ids {
            // Preempted may already be removed — take_finished handles missing
            if let Some(seq) = self.scheduler.take_finished(id) {
                self.results.insert(id, seq.tokens);
                self.runtimes.remove(&id);
                self.metrics.finished += 1;
                done.push(id);
            } else if let Some(tokens) = self.results.get(&id) {
                let _ = tokens;
                done.push(id);
            }
        }
        Ok(done)
    }

    pub fn run_all(&mut self) -> Result<HashMap<u64, Vec<u32>>> {
        while self.has_unfinished() {
            let _ = self.step()?;
        }
        tracing::info!(
            steps = self.metrics.steps,
            prefills = self.metrics.prefills,
            decodes = self.metrics.decodes,
            tesla = self.metrics.tesla_steps,
            finished = self.metrics.finished,
            preempted = self.metrics.preempted,
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

    fn run_prefill(&mut self, id: u64) -> Result<()> {
        let (prompt, sampling, max_tokens, prompt_len) = {
            let seq = self
                .scheduler
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("missing seq {id}"))?;
            (
                seq.tokens[..seq.prompt_len].to_vec(),
                seq.sampling.clone(),
                seq.stop.max_tokens,
                seq.prompt_len,
            )
        };

        let rt = self
            .runtimes
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("missing runtime {id}"))?;

        let device = &self.engine.device;
        let prefill = Tensor::new(prompt.as_slice(), device)?.unsqueeze(0)?;
        let logits = squeeze_logits(&self.engine.model.forward(
            &prefill,
            0,
            Some(&mut rt.caches),
        )?)?;
        let next = sample_token(&logits, &sampling, &rt.generated, &mut rt.rng)?;
        rt.generated.push(next);
        self.metrics.tokens_generated += 1;

        if let Some(seq) = self.scheduler.get_mut(id) {
            seq.append_token(next);
            seq.mark_computed(prompt_len);
            if is_stop(next, &sampling) || seq.generated_len() >= max_tokens {
                seq.finish(finish_status(seq.generated_len(), max_tokens, next, &sampling));
            }
        }
        Ok(())
    }

    fn run_decode_plain(&mut self, id: u64) -> Result<()> {
        let (sampling, max_tokens, last_token) = {
            let seq = self
                .scheduler
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("missing seq {id}"))?;
            let last = *seq.tokens.last().unwrap_or(&0);
            (seq.sampling.clone(), seq.stop.max_tokens, last)
        };

        let rt = self
            .runtimes
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("missing runtime {id}"))?;
        let device = &self.engine.device;
        let pos = rt.caches.current_len();
        let input = Tensor::new(&[last_token], device)?.unsqueeze(0)?;
        let logits = squeeze_logits(&self.engine.model.forward(
            &input,
            pos,
            Some(&mut rt.caches),
        )?)?;
        let next = sample_token(&logits, &sampling, &rt.generated, &mut rt.rng)?;
        rt.generated.push(next);
        self.metrics.tokens_generated += 1;

        if let Some(seq) = self.scheduler.get_mut(id) {
            seq.append_token(next);
            seq.mark_computed(1);
            if is_stop(next, &sampling) || seq.generated_len() >= max_tokens {
                seq.finish(finish_status(seq.generated_len(), max_tokens, next, &sampling));
            }
        }
        Ok(())
    }

    fn run_decode_tesla(&mut self, id: u64) -> Result<()> {
        // Borrow tesla config without cloning whole engine
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

        tracing::debug!(
            seq = id,
            drafted = stats.drafted,
            scheduled = stats.scheduled,
            accepted = stats.accepted,
            accept_rate = rt.mtp.accept_rate,
            forwards = stats.forwards,
            "Tesla MTP step"
        );

        for &t in &accepted {
            rt.generated.push(t);
            self.metrics.tokens_generated += 1;
            if let Some(seq) = self.scheduler.get_mut(id) {
                seq.append_token(t);
                seq.mark_computed(1);
                if is_stop(t, &sampling) || seq.generated_len() >= max_tokens {
                    seq.finish(finish_status(seq.generated_len(), max_tokens, t, &sampling));
                    break;
                }
            }
        }
        Ok(())
    }
}

fn finish_status(
    generated_len: usize,
    max_tokens: usize,
    last: u32,
    sampling: &SamplingParams,
) -> SequenceStatus {
    if generated_len >= max_tokens {
        SequenceStatus::FinishedLengthCapped
    } else if is_stop(last, sampling) {
        SequenceStatus::FinishedStopped
    } else {
        SequenceStatus::FinishedStopped
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
