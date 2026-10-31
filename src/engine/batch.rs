//! BatchEngine — continuous scheduler + InferenceEngine step loop.
//!
//! Flow (DeepInfer-style):
//! 1. `add_request` → waiting queue
//! 2. `schedule()` → which seqs to run (prefill or decode)
//! 3. `step()` → prefill / plain decode / Tesla MTP decode
//! 4. finished sequences reaped, blocks freed, tokens stored in `results`

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
    next_token: Option<u32>,
    rng: StdRng,
}

pub struct BatchEngine {
    engine: InferenceEngine,
    scheduler: ContinuousScheduler,
    tesla: Option<TeslaConfig>,
    runtimes: HashMap<u64, SeqRuntime>,
    /// Completed request outputs: seq_id → full token ids (prompt + gen).
    pub results: HashMap<u64, Vec<u32>>,
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

        let scheduled = outputs.scheduled.clone();

        for &id in &scheduled {
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
        }

        // Reap anything marked finished during this step
        let mut finished = outputs.finished;
        for id in self.scheduler.running_ids() {
            if self.scheduler.get(id).map(|s| s.is_finished()).unwrap_or(false) {
                finished.push(id);
            }
        }
        finished.sort_unstable();
        finished.dedup();

        let mut done = Vec::new();
        for id in finished {
            if let Some(seq) = self.scheduler.take_finished(id) {
                self.results.insert(id, seq.tokens);
                self.runtimes.remove(&id);
                done.push(id);
            }
        }
        Ok(done)
    }

    /// Run until all sequences finish. Returns map id → full token ids.
    pub fn run_all(&mut self) -> Result<HashMap<u64, Vec<u32>>> {
        while self.has_unfinished() {
            let _ = self.step()?;
        }
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
