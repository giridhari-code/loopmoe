//! Custom CUDA kernels for Sentinel (Fertile / DoBu).
//!
//! Built from `sentinel.cu` via `nvcc -ptx` in `build.rs` when the `cuda`
//! feature is on and the toolkit is available. Otherwise all ops fall back
//! to portable Candle implementations.

mod ops;
mod gemm;

pub use ops::{bias_add, rmsnorm, scale_mul, scatter_add_rows, silu_mul, kernels_available};
pub use gemm::{tiled_gemm_nn, tiled_gemm_nt, should_use_tiled};

include!(concat!(env!("OUT_DIR"), "/sentinel_ptx.rs"));

pub fn cuda_available() -> bool {
    candle_core::Device::cuda_if_available(0)
        .map(|d| d.is_cuda())
        .unwrap_or(false)
}

pub fn kernels_built() -> bool {
    SENTINEL_KERNELS_BUILT && !SENTINEL_PTX.is_empty()
}
