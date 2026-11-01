//! Custom CUDA kernels for Sentinel (Fertile / DoBu / FP8 / fused norms).

mod ops;
mod gemm;
mod fp8;

pub use ops::{
    bias_add, kernels_available, residual_add, rmsnorm, scale_mul, scatter_add_rows, silu_mul,
};
pub use gemm::{should_use_tiled, tiled_gemm_nn, tiled_gemm_nt};
pub use fp8::{
    dequant_fp8_e4m3, fp8_scale_from_amax, fp8_supported, gemm_nn_fp8_weight, quantize_to_fp8_e4m3,
};

include!(concat!(env!("OUT_DIR"), "/sentinel_ptx.rs"));

pub fn cuda_available() -> bool {
    candle_core::Device::cuda_if_available(0)
        .map(|d| d.is_cuda())
        .unwrap_or(false)
}

pub fn kernels_built() -> bool {
    SENTINEL_KERNELS_BUILT && !SENTINEL_PTX.is_empty()
}

pub fn fp8_kernels_built() -> bool {
    SENTINEL_FP8_BUILT && !SENTINEL_FP8_PTX.is_empty()
}
