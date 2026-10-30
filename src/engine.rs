use candle_core::{Device, Tensor, DType};
use candle_nn::VarBuilder;
use safetensors::SafeTensors;
use std::path::Path;
use crate::{config::ModelConfig, model::LoopedMoE, Result};

pub struct InferenceEngine {
    pub model: LoopedMoE,
    pub device: Device,
    pub config: ModelConfig,
}

impl InferenceEngine {
    /// Load model from safetensors directory or single file.
    /// Tries CUDA first, falls back to CPU.
    pub fn from_safetensors<P: AsRef<Path>>(
        model_path: P,
        config: ModelConfig,
    ) -> Result<Self> {
        let device = select_device()?;
        tracing::info!("Using device: {:?}", device);

        // Load weights
        let weights_path = model_path.as_ref();
        let vb = if weights_path.is_dir() {
            // multi-file safetensors (common for large models)
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

    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>> {
        let mut tokens = input_ids.to_vec();
        let mut start_pos = 0;

        for _ in 0..max_new_tokens {
            let input = Tensor::new(&tokens[start_pos..], &self.device)?
                .unsqueeze(0)?; // [1, seq]

            let logits = self.model.forward(&input, start_pos)?;
            // greedy for now
            let next = logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?;
            tokens.push(next);
            start_pos = tokens.len() - 1;

            // TODO: EOS handling
        }
        Ok(tokens)
    }
}

fn select_device() -> Result<Device> {
    // GPU first
    if let Ok(device) = Device::new_cuda(0) {
        return Ok(device);
    }
    // fallback
    tracing::warn!("CUDA not available, falling back to CPU");
    Ok(Device::Cpu)
}
