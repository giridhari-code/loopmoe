use candle_core::{Device, Tensor, DType, IndexOp};
use std::path::Path;
use crate::{
    config::ModelConfig,
    model::LoopedMoE,
    cache::{kv_bytes_estimate, LayerCaches, MlaCaches, RuntimeCaches},
    kernels::BackendInfo,
    sampler::{sample_token, is_stop, SamplingParams},
    tesla::{draft_tokens, schedule_verify_len, verify_and_accept, TeslaConfig, TeslaStepStats},
    Result,
};
use rand::{rngs::StdRng, SeedableRng};

pub struct InferenceEngine {
    pub model: LoopedMoE,
    pub device: Device,
    pub config: ModelConfig,
    dtype: DType,
}

impl InferenceEngine {
    pub fn from_safetensors<P: AsRef<Path>>(model_path: P, config: ModelConfig) -> Result<Self> {
        let device = select_device()?;
        tracing::info!("Using device: {:?}", device);
        let backend = BackendInfo::from_device(&device);
        tracing::info!(fertile_backend = %backend.kind, "Fertile kernel backend");
        let dtype = DType::BF16;
        let weights_path = model_path.as_ref();
        let vb = if weights_path.is_dir() {
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path.join("model.safetensors")], dtype, &device)? }
        } else {
            unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&[weights_path.to_path_buf()], dtype, &device)? }
        };
        let model = LoopedMoE::load(vb, &config, &device)?;
        let kv_est = kv_bytes_estimate(config.num_layers, 1, config.num_key_value_heads, config.head_dim, config.max_seq_len, 2);
        tracing::info!(max_seq = config.max_seq_len, kv_cache_mib = kv_est / (1024 * 1024), "RAM plan");
        Ok(Self { model, device, config, dtype })
    }

    pub fn generate(&self, input_ids: &[u32], max_new_tokens: usize, params: &SamplingParams) -> Result<Vec<u32>> {
        self.generate_inner(input_ids, max_new_tokens, params, None)
    }

    pub fn generate_tesla(&self, input_ids: &[u32], max_new_tokens: usize, params: &SamplingParams, tesla: &TeslaConfig) -> Result<Vec<u32>> {
        if !tesla.enabled { return self.generate(input_ids, max_new_tokens, params); }
        self.generate_inner(input_ids, max_new_tokens, params, Some(tesla))
    }

    fn generate_inner(&self, input_ids: &[u32], max_new_tokens: usize, params: &SamplingParams, tesla: Option<&TeslaConfig>) -> Result<Vec<u32>> {
        if input_ids.is_empty() { anyhow::bail!("empty prompt"); }
        let prompt_len = input_ids.len();
        let capacity = (prompt_len + max_new_tokens).min(self.config.max_seq_len).max(prompt_len + 1);
        let mut caches = if self.config.use_mla {
            RuntimeCaches::Mla(MlaCaches::new(self.config.num_layers))
        } else {
            RuntimeCaches::Gqa(LayerCaches::with_capacity(self.config.num_layers, 1, self.config.num_key_value_heads, self.config.head_dim, capacity, self.dtype, &self.device))
        };
        let mut rng = match params.seed { Some(s) => StdRng::seed_from_u64(s), None => StdRng::from_entropy() };
        let mut tokens = input_ids.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut total_stats = TeslaStepStats::default();

        let prefill = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits = squeeze_logits(&self.model.forward(&prefill, 0, Some(&mut caches))?)?;
        let mut next = sample_token(&logits, params, &generated, &mut rng)?;
        tokens.push(next); generated.push(next);
        if is_stop(next, params) { caches.clear(); return Ok(tokens); }

        while generated.len() < max_new_tokens {
            if caches.current_len() >= self.config.max_seq_len { break; }
            if let Some(cfg) = tesla {
                let pos = caches.current_len();
                let lens_ckpt = caches.checkpoint_lens();
                let draft = draft_tokens(&self.model, &mut caches, next, pos, &self.device, cfg, params, &generated, &mut rng)?;
                let vlen = schedule_verify_len(&draft.confidences, cfg.max_draft, cfg.min_confidence, cfg.min_survival);
                let (accepted, stats) = verify_and_accept(&self.model, &mut caches, &lens_ckpt, next, pos, &draft.tokens, vlen, &self.device, params, &generated, &mut rng)?;
                total_stats.drafted += stats.drafted;
                total_stats.scheduled += stats.scheduled;
                total_stats.accepted += stats.accepted;
                for &t in &accepted {
                    tokens.push(t); generated.push(t); next = t;
                    if is_stop(t, params) || generated.len() >= max_new_tokens {
                        caches.clear();
                        tracing::info!(drafted = total_stats.drafted, scheduled = total_stats.scheduled, accepted = total_stats.accepted, "Tesla stats");
                        return Ok(tokens);
                    }
                }
            } else {
                let start_pos = caches.current_len();
                let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
                let logits = squeeze_logits(&self.model.forward(&input, start_pos, Some(&mut caches))?)?;
                next = sample_token(&logits, params, &generated, &mut rng)?;
                tokens.push(next); generated.push(next);
                if is_stop(next, params) { break; }
            }
        }
        if tesla.is_some() {
            tracing::info!(drafted = total_stats.drafted, scheduled = total_stats.scheduled, accepted = total_stats.accepted, "Tesla stats");
        }
        caches.clear();
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

fn select_device() -> Result<Device> {
    if let Ok(device) = Device::new_cuda(0) { return Ok(device); }
    tracing::warn!("CUDA not available, falling back to CPU/ARM");
    Ok(Device::Cpu)
}
