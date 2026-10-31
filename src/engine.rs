use candle_core::{Device, Tensor, DType, IndexOp};
use std::path::Path;
use crate::{
    config::ModelConfig,
    model::LoopedMoE,
    cache::{
        kv_bytes_estimate, mla_bytes_estimate, LayerCaches, MlaCaches, RuntimeCaches,
    },
    kernels::BackendInfo,
    sampler::{sample_token, is_stop, SamplingParams},
    tesla::{
        draft_tokens, schedule_verify_len, schedule_verify_len_smart,
        verify_and_accept, TeslaConfig, TeslaStepStats,
    },
    parallel::{DeviceMesh, ExpertParallelPlan, ParallelConfig},
    Result,
};
use rand::{rngs::StdRng, SeedableRng};

pub struct InferenceEngine {
    pub model: LoopedMoE,
    pub device: Device,
    pub config: ModelConfig,
    pub mesh: DeviceMesh,
    pub ep_plan: ExpertParallelPlan,
    dtype: DType,
}

impl InferenceEngine {
    pub fn from_safetensors<P: AsRef<Path>>(model_path: P, config: ModelConfig) -> Result<Self> {
        Self::from_safetensors_parallel(model_path, config, ParallelConfig::single())
    }

    pub fn from_safetensors_parallel<P: AsRef<Path>>(
        model_path: P,
        config: ModelConfig,
        parallel: ParallelConfig,
    ) -> Result<Self> {
        parallel.validate()?;
        let device = DeviceMesh::select_device(parallel.local_rank)?;
        let mesh = DeviceMesh::from_config(parallel, device.clone())?;
        tracing::info!(
            device = ?mesh.device(),
            world = mesh.config.world_size,
            tp = mesh.config.tp_size,
            ep = mesh.config.ep_size,
            dp = mesh.config.dp_size,
            kind = ?mesh.config.kind(),
            "Device mesh"
        );
        let backend = BackendInfo::from_device(mesh.device());
        tracing::info!(fertile_backend = %backend.kind, "Fertile kernel backend");

        let dtype = DType::BF16;
        let weights_path = model_path.as_ref();
        let vb = if weights_path.is_dir() {
            unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_path.join("model.safetensors")],
                    dtype,
                    mesh.device(),
                )?
            }
        } else {
            unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_path.to_path_buf()],
                    dtype,
                    mesh.device(),
                )?
            }
        };
        let model = LoopedMoE::load(vb, &config, mesh.device())?;

        let ep_plan = if mesh.is_single() {
            ExpertParallelPlan::single(
                config.n_routed_experts,
                config.n_shared_experts,
                config.num_experts_per_tok,
            )
        } else {
            ExpertParallelPlan::from_mesh(
                &mesh,
                config.n_routed_experts,
                config.n_shared_experts,
                config.num_experts_per_tok,
                config.capacity_factor,
            )?
        };
        tracing::info!(
            ep_rank = ep_plan.shard.ep_rank,
            local_experts = ep_plan.local_routed_count(),
            shared = ep_plan.n_shared_experts,
            capacity_factor = ep_plan.capacity_factor,
            "Expert parallel plan"
        );

        if config.use_mla {
            let mla_est = mla_bytes_estimate(
                config.num_layers,
                1,
                config.kv_lora_rank,
                config.qk_rope_head_dim,
                config.max_seq_len,
                2, // BF16
            );
            tracing::info!(
                max_seq = config.max_seq_len,
                mla_cache_mib = mla_est / (1024 * 1024),
                rank = config.kv_lora_rank,
                "MLA RAM plan (prealloc)"
            );
        } else {
            let kv_est = kv_bytes_estimate(
                config.num_layers,
                1,
                config.num_key_value_heads,
                config.head_dim,
                config.max_seq_len,
                2,
            );
            tracing::info!(
                max_seq = config.max_seq_len,
                kv_cache_mib = kv_est / (1024 * 1024),
                "GQA RAM plan (prealloc)"
            );
        }

        Ok(Self {
            model,
            device: mesh.device().clone(),
            config,
            mesh,
            ep_plan,
            dtype,
        })
    }

    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
    ) -> Result<Vec<u32>> {
        self.generate_inner(input_ids, max_new_tokens, params, None)
    }

    pub fn generate_tesla(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        tesla: &TeslaConfig,
    ) -> Result<Vec<u32>> {
        if !tesla.enabled {
            return self.generate(input_ids, max_new_tokens, params);
        }
        self.generate_inner(input_ids, max_new_tokens, params, Some(tesla))
    }

    fn generate_inner(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
        params: &SamplingParams,
        tesla: Option<&TeslaConfig>,
    ) -> Result<Vec<u32>> {
        if input_ids.is_empty() {
            anyhow::bail!("empty prompt");
        }
        let prompt_len = input_ids.len();
        let capacity = (prompt_len + max_new_tokens)
            .min(self.config.max_seq_len)
            .max(prompt_len + 1);

        let mut caches = if self.config.use_mla {
            RuntimeCaches::Mla(MlaCaches::with_capacity(
                self.config.num_layers,
                1,
                self.config.kv_lora_rank,
                self.config.qk_rope_head_dim,
                capacity,
                self.dtype,
                &self.device,
            ))
        } else {
            RuntimeCaches::Gqa(LayerCaches::with_capacity(
                self.config.num_layers,
                1,
                self.config.num_key_value_heads,
                self.config.head_dim,
                capacity,
                self.dtype,
                &self.device,
            ))
        };

        let mut rng = match params.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let mut tokens = input_ids.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut total_stats = TeslaStepStats::default();

        let prefill = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits = squeeze_logits(&self.model.forward(&prefill, 0, Some(&mut caches))?)?;
        let mut next = sample_token(&logits, params, &generated, &mut rng)?;
        tokens.push(next);
        generated.push(next);
        if is_stop(next, params) {
            caches.release();
            return Ok(tokens);
        }

        while generated.len() < max_new_tokens {
            if caches.current_len() >= self.config.max_seq_len {
                break;
            }

            if let Some(cfg) = tesla {
                let pos = caches.current_len();
                let lens_ckpt = caches.checkpoint_lens();

                let draft = draft_tokens(
                    &self.model,
                    &mut caches,
                    next,
                    pos,
                    &self.device,
                    cfg,
                    params,
                    &generated,
                    &mut rng,
                )?;

                let vlen = if cfg.use_margin {
                    schedule_verify_len_smart(
                        &draft.confidences,
                        Some(&draft.margins),
                        cfg.max_draft,
                        cfg.min_confidence,
                        cfg.min_survival,
                        cfg.min_margin,
                    )
                } else {
                    schedule_verify_len(
                        &draft.confidences,
                        cfg.max_draft,
                        cfg.min_confidence,
                        cfg.min_survival,
                    )
                };

                let (accepted, stats) = verify_and_accept(
                    &self.model,
                    &mut caches,
                    &lens_ckpt,
                    next,
                    pos,
                    &draft.tokens,
                    vlen,
                    &self.device,
                    params,
                    &generated,
                    &mut rng,
                )?;

                total_stats.drafted += stats.drafted;
                total_stats.scheduled += stats.scheduled;
                total_stats.accepted += stats.accepted;
                total_stats.forwards += stats.forwards;

                for &t in &accepted {
                    tokens.push(t);
                    generated.push(t);
                    next = t;
                    if is_stop(t, params) || generated.len() >= max_new_tokens {
                        caches.release();
                        tracing::info!(
                            drafted = total_stats.drafted,
                            scheduled = total_stats.scheduled,
                            accepted = total_stats.accepted,
                            forwards = total_stats.forwards,
                            accept_rate = total_stats.acceptance_rate(),
                            "Tesla stats (chunk verify)"
                        );
                        return Ok(tokens);
                    }
                }
            } else {
                let start_pos = caches.current_len();
                let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                let logits =
                    squeeze_logits(&self.model.forward(&input, start_pos, Some(&mut caches))?)?;
                next = sample_token(&logits, params, &generated, &mut rng)?;
                tokens.push(next);
                generated.push(next);
                if is_stop(next, params) {
                    break;
                }
            }
        }

        if tesla.is_some() {
            tracing::info!(
                drafted = total_stats.drafted,
                scheduled = total_stats.scheduled,
                accepted = total_stats.accepted,
                forwards = total_stats.forwards,
                accept_rate = total_stats.acceptance_rate(),
                "Tesla stats (chunk verify)"
            );
        }
        caches.release();
        Ok(tokens)
    }

    pub fn forward_logits(&self, input_ids: &[u32]) -> Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        Ok(self.model.forward(&input, 0, None)?)
    }
}

fn squeeze_logits(logits: &Tensor) -> candle_core::Result<Tensor> {
    match logits.dims() {
        [_] => Ok(logits.clone()),
        [1, _] => logits.squeeze(0),
        [1, 1, _] => logits.squeeze(0)?.squeeze(0),
        dims if dims.len() == 2 => logits.i(0),
        _ => logits.flatten_all(),
    }
}
