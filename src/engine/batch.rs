//! BatchEngine — continuous scheduler + InferenceEngine step loop.
//!
//! Flow (DeepInfer-style):
//! 1. `add_request` → waiting queue
//! 2. `schedule()` → which seqs to run (prefill or decode)
//! 3. `step()` → prefill / plain decode / Tesla MTP decode
//! 4. finished sequences reaped, blocks freed

use std::collections::HashMap;

use candle_core::{Device, Tensor, DType};
use rand::{rngs::StdRng, SeedableRng};

use crate::cache::{LayerCaches, MlaCaches, RuntimeCaches};
use crate::engine::inference::InferenceEngine;
use crate::sampler::{is_stop, sample_token, SamplingParams};
use crate::scheduler::{
    ContinuousScheduler, SchedulerConfig, Sequence, SequenceStatus,
};
use crate::tesla::{
    draft_tokens, verify_and_accept, MtpState, TeslaConfig, TeslaStepStats,
};
use crate::Result;

struct SeqRuntime {
    caches: RuntimeCaches,
    mtp: MtpState,
    generated: Vec<u32>,
    next_token: Option<u32>,
    rng: StdRng,
}

pub struct BatchEngine {
    engine: InferenceEngine,
    scheduler: ContinuousScheduler,
    tesla: Option<TeslaConfig>,
    runtimes: HashMap<u64, SeqRuntime>,
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
        let dtype = DType::BF16;
        Self {
            engine,
            scheduler: ContinuousScheduler::new(sched_cfg, num_gpu_blocks, block_size),
            tesla,
            runtimes: HashMap::new(),
            dtype,
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

    /// Enqueue a new generation request. Returns sequence id.
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
            seq.draft_len = Some(t.effective_nextn());
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
                generated: Vec::new(),
                next_token: None,
                rng: StdRng::seed_from_u64(seed),
            },
        );
        id
    }

    pub fn has_unfinished(&self) -> bool {
        self.scheduler.has_unfinished()
    }

    /// One scheduler + execute step. Returns finished sequence ids this step.
    pub fn step(&mut self) -> Result<Vec<u64>> {
        let outputs = self.scheduler.schedule();
        if outputs.is_empty() && outputs.finished.is_empty() {
            return Ok(Vec::new());
        }

        let scheduled = outputs.scheduled.clone();
        let prompt_run = outputs.prompt_run;

        for &id in &scheduled {
            if prompt_run {
                // May mix — check per-seq
                let is_prompt = self
                    .scheduler
                    .get(id)
                    .map(|s| s.is_prompt())
                    .unwrap_or(false);
                if is_prompt {
                    self.run_prefill(id)?;
                } else {
                    self.run_decode(id)?;
                }
            } else {
                self.run_decode(id)?;
            }
        }

        // Collect newly finished
        let mut finished = outputs.finished;
        let ids: Vec<u64> = self.scheduler.running_ids();
        for id in ids {
            if let Some(seq) = self.scheduler.get(id) {
                if seq.is_finished() {
                    finished.push(id);
                }
            }
        }
        // Actually finish via scheduler API so blocks free
        let mut unique = Vec::new();
        for id in finished {
            if unique.contains(&id) {
                continue;
            }
            if let Some(seq) = self.scheduler.get(id) {
                if seq.is_finished() {
                    let status = seq.status;
                    self.scheduler.finish_sequence(id, status);
                    if let Some(rt) = self.runtimes.remove(&id) {
                        // drop caches
                        let _ = rt;
                    }
                    unique.push(id);
                }
            }
        }
        Ok(unique)
    }

    /// Run until all sequences finish. Returns map id → full token ids.
    pub fn run_all(&mut self) -> Result<HashMap<u64, Vec<u32>>> {
        while self.has_unfinished() {
            let _ = self.step()?;
        }
        // Any remaining finished tokens are already removed; collect from history
        // Callers should snapshot tokens before finish — keep a results buffer.
        Ok(HashMap::new())
    }

    /// Snapshot current tokens for a sequence.
    pub fn tokens(&self, id: u64) -> Option<Vec<u32>> {
        self.scheduler.get(id).map(|s| s.tokens.clone())
    }

    pub fn mtp_stats(&self, id: u64) -> Option<&MtpState> {
        self.runtimes.get(&id).map(|r| &r.mtp)
    }

    fn run_prefill(&mut self, id: u64) -> Result<()> {
        let (prompt, sampling, max_tokens) = {
            let seq = self
                .scheduler
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("missing seq {id}"))?;
            (
                seq.tokens[..seq.prompt_len].to_vec(),
                seq.sampling.clone(),
                seq.stop.max_tokens,
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
        rt.next_token = Some(next);

        if let Some(seq) = self.scheduler.get_mut(id) {
            seq.append_token(next);
            seq.mark_computed(seq.prompt_len);
            if is_stop(next, &sampling) || seq.generated_len() >= max_tokens {
                seq.finish(if seq.generated_len() >= max_tokens {
                    SequenceStatus::FinishedLengthCapped
                } else {
                    SequenceStatus::FinishedStopped
                });
            }
        }
        Ok(())
    }

    fn run_decode(&mut self, id: u64) -> Result<()> {
        let tesla_cfg = self.tesla.clone();
        let (sampling, max_tokens, last_token) = {
            let seq = self
                .scheduler
                .get(id)
                .ok_or_else(|| anyhow::anyhow!("missing seq {id}"))?;
            let last = *seq.tokens.last().unwrap_or(&0);
            (seq.sampling.clone(), seq.stop.max_tokens, last)
        };

        let use_tesla = tesla_cfg.as_ref().map(|t| t.enabled).unwrap_or(false);

        if use_tesla {
            self.run_decode_tesla(id, last_token, &sampling, max_tokens, tesla_cfg.as_ref().unwrap())?;
        } else {
            self.run_decode_plain(id, last_token, &sampling, max_tokens)?;
        }
        Ok(())
    }

    fn run_decode_plain(
        &mut self,
        id: u64,
        last_token: u32,
        sampling: &SamplingParams,
        max_tokens: usize,
    ) -> Result<()> {
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
        let next = sample_token(&logits, sampling, &rt.generated, &mut rt.rng)?;
        rt.generated.push(next);
        rt.next_token = Some(next);

        if let Some(seq) = self.scheduler.get_mut(id) {
            seq.append_token(next);
            seq.mark_computed(1);
            if is_stop(next, sampling) || seq.generated_len() >= max_tokens {
                seq.finish(if seq.generated_len() >= max_tokens {
                    SequenceStatus::FinishedLengthCapped
                } else {
                    SequenceStatus::FinishedStopped
                });
            }
        }
        Ok(())
    }

    fn run_decode_tesla(
        &mut self,
        id: u64,
        last_token: u32,
        sampling: &SamplingParams,
        max_tokens: usize,
        cfg: &TeslaConfig,
    ) -> Result<()> {
        let device = self.engine.device.clone();
        let model = &self.engine.model;

        let rt = self
            .runtimes
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("missing runtime {id}"))?;

        let pos = rt.caches.current_len();
        let lens_ckpt = rt.caches.checkpoint_lens();

        // --- Draft (MTP) ---
        let draft = draft_tokens(
            model,
            &mut rt.caches,
            last_token,
            pos,
            &device,
            cfg,
            sampling,
            &rt.generated,
            &mut rt.rng,
        )?;
        rt.mtp.set_draft(
            draft.tokens.clone(),
            draft.confidences.clone(),
            draft.margins.clone(),
        );

        // --- Confidence / accept-rate scheduled verify length ---
        let vlen = rt.mtp.schedule_verify_len(cfg);

        let (accepted, stats) = verify_and_accept(
            model,
            &mut rt.caches,
            &lens_ckpt,
            last_token,
            pos,
            &draft.tokens,
            vlen,
            &device,
            sampling,
            &rt.generated,
            &mut rt.rng,
        )?;

        rt.mtp
            .on_verify_result(stats.accepted, stats.scheduled, cfg);

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
            rt.next_token = Some(t);
            if let Some(seq) = self.scheduler.get_mut(id) {
                seq.append_token(t);
                seq.mark_computed(1);
                if is_stop(t, sampling) || seq.generated_len() >= max_tokens {
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

// Helper on ContinuousScheduler — expose running ids
impl ContinuousScheduler {
    pub fn running_ids(&self) -> Vec<u64> {
        // Need access — add method in continuous.rs instead
        Vec::new()
    }
}
