use candle_core::{Device, Tensor, DType, IndexOp};
use std::path::Path;
use crate::{
    config::ModelConfig,
    model::LoopedMoE,
    cache::{kv_bytes_estimate, LayerCaches},
    kernels::BackendInfo,
    sampler::{sample_token, is_stop, SamplingParams},
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
            unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_path.join("model.safetensors")], dtype, &device,
                )?
            }
        } else {
            unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_path.to_path_buf()], dtype, &device,
                )?
            }
        };
        let model = LoopedMoE::load(vb, &config, &device)?;
        let kv_est = kv_bytes_estimate(
            config.num_layers, 1, config.num_key_value_heads, config.head_dim, config.max_seq_len, 2,
        );
        tracing::info!(max_seq = config.max_seq_len, kv_cache_mib = kv_est / (1024 * 1024), "RAM plan");
        Ok(Self { model, device, config, dtype })
    }

    pub fn generate(&self, input_ids: &[u32], max_new_tokens: usize, params: &SamplingParams) -> Result<Vec<u32>> {
        if input_ids.is_empty() { anyhow::bail!("empty prompt"); }
        let prompt_len = input_ids.len();
        let capacity = (prompt_len + max_new_tokens).min(self.config.max_seq_len).max(prompt_len + 1);
        let mut caches = LayerCaches::with_capacity(
            self.config.num_layers, 1, self.config.num_key_value_heads, self.config.head_dim,
            capacity, self.dtype, &self.device,
        );
        let mut rng = match params.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let mut tokens = input_ids.to_vec();
        let mut generated: Vec<u32> = Vec::new();

        let prefill = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits = squeeze_logits(&self.model.forward(&prefill, 0, Some(&mut caches))?)?;
        let mut next = sample_token(&logits, params, &generated, &mut rng)?;
        tokens.push(next);
        generated.push(next);
        if is_stop(next, params) { caches.clear(); return Ok(tokens); }

        for _ in 1..max_new_tokens {
            if caches.current_len() >= self.config.max_seq_len { break; }
            let start_pos = caches.current_len();
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let logits = squeeze_logits(&self.model.forward(&input, start_pos, Some(&mut caches))?)?;
            next = sample_token(&logits, params, &generated, &mut rng)?;
            tokens.push(next);
            generated.push(next);
            if is_stop(next, params) { break; }
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
