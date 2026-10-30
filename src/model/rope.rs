//! Rotary Position Embedding (RoPE)
use candle_core::{D, DType, Device, Result as CandleResult, Tensor};

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    pub head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq: usize,
        theta: f32,
        device: &Device,
        dtype: DType,
    ) -> CandleResult<Self> {
        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let freq = 1.0_f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
            inv_freq.push(freq);
        }
        let inv_freq = Tensor::new(inv_freq.as_slice(), device)?;

        let t = Tensor::arange(0u32, max_seq as u32, device)?.to_dtype(DType::F32)?;
        let freqs = t.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        let emb = Tensor::cat(&[&freqs, &freqs], 1)?;
        let cos = emb.cos()?.to_dtype(dtype)?;
        let sin = emb.sin()?.to_dtype(dtype)?;

        Ok(Self { sin, cos, head_dim })
    }

    pub fn forward(&self, x: &Tensor, start_pos: usize) -> CandleResult<Tensor> {
        let (_b, _h, s, _d) = x.dims4()?;
        let cos = self.cos.narrow(0, start_pos, s)?.unsqueeze(0)?.unsqueeze(0)?;
        let sin = self.sin.narrow(0, start_pos, s)?.unsqueeze(0)?.unsqueeze(0)?;

        let x_rot = rotate_half(x)?;
        x.broadcast_mul(&cos)? + x_rot.broadcast_mul(&sin)?
    }
}

fn rotate_half(x: &Tensor) -> CandleResult<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}
