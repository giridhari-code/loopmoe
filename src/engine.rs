use candle_core::{Device, Tensor, DType, D};
use std::path::Path;
use crate::{config::ModelConfig, model::LoopedMoE, cache::LayerCaches, Result};

pub struct InferenceEngine {
    pub model: LoopedMoE,
    pub device: Device,
    pub config: ModelConfig,
}

impl InferenceEngine {
    pub fn from_safetensors<P: AsRef<Path>>(
        model_path: P,
        config: ModelConfig,
    ) -> Result<Self> {
        let device = select_device()?;
        tracing::info!("Using device: {:?}", device);

        let weights_path = model_path.as_ref();
        let vb = if weights_path.is_dir() {
            unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_path.join("model.safetensors")],
                    DType::BF16,
                    &device,
                )?
            }
        } else {
            unsafe {
                candle_nn::VarBuilder::from_mmaped_safetensors(
                    &[weights_path.to_path_buf()],
                    DType::BF16,
                    &device,
                )?
            }
        };

        let model = LoopedMoE::load(vb, &config, &device)?;
        Ok(Self { model, device, config })
    }

    /// Prefill then decode with KV cache.
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();
        let mut caches = LayerCaches::new(self.config.num_layers);

        // Prefill
        let prefill = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&prefill, 0, Some(&mut caches))?;
        let mut next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        tokens.push(next);

        // Decode
        for _ in 1..max_new_tokens {
            let start_pos = caches.current_len();
            let input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos, Some(&mut caches))?;
            next = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            tokens.push(next);
        }
        Ok(tokens)
    }

    pub fn forward_logits(&self, input_ids: &[u32]) -> Result<Tensor> {
        let input = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        Ok(self.model.forward(&input, 0, None)?)
    }
}

fn select_device() -> Result<Device> {
    if let Ok(device) = Device::new_cuda(0) {
        return Ok(device);
    }
    tracing::warn!("CUDA not available, falling back to CPU");
    Ok(Device::Cpu)
}
