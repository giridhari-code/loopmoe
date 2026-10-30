//! CUDA helpers for Fertile.

use candle_core::Device;

pub fn cuda_available() -> bool {
    Device::cuda_if_available(0).is_cuda()
}
