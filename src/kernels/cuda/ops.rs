//! High-level wrappers around Sentinel custom CUDA kernels.
//!
//! On CUDA + built PTX → launch custom kernel.
//! Otherwise → equivalent Candle ops (always correct).

use candle_core::{
    backend::BackendStorage, CpuStorage, CustomOp1, CustomOp2, DType, Layout, Result as CResult,
    Shape, Storage, Tensor,
};

use super::{SENTINEL_KERNELS_BUILT, SENTINEL_PTX};

pub fn kernels_available(t: &Tensor) -> bool {
    SENTINEL_KERNELS_BUILT && t.device().is_cuda() && t.dtype() == DType::F32
}

// ---------------------------------------------------------------------------
// silu(gate) * up
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SiluMulOp;

impl CustomOp2 for SiluMulOp {
    fn name(&self) -> &'static str {
        "sentinel_silu_mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        let (gate, shape) = s1.as_slice::<f32>()?.as_ref();
        // Layout-aware contiguous views
        let g = match s1 {
            CpuStorage::F32(v) => v,
            _ => candle_core::bail!("silu_mul cpu: expected f32"),
        };
        let u = match s2 {
            CpuStorage::F32(v) => v,
            _ => candle_core::bail!("silu_mul cpu: expected f32"),
        };
        let n = l1.shape().elem_count();
        let start1 = l1.start_offset();
        let start2 = l2.start_offset();
        let mut out = vec![0f32; n];
        for i in 0..n {
            let x = g[start1 + i];
            let s = 1.0 / (1.0 + (-x).exp());
            out[i] = (x * s) * u[start2 + i];
        }
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> CResult<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if !(SENTINEL_KERNELS_BUILT && !SENTINEL_PTX.is_empty()) {
            // Fall through — caller should not reach here without kernels;
            // still provide a safe error.
            candle_core::bail!("sentinel CUDA kernels not built; rebuild with nvcc");
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("sentinel_silu_mul requires contiguous tensors");
        }
        let dev = s1.device();
        let slice1 = s1.as_cuda_slice::<f32>()?;
        let slice2 = s2.as_cuda_slice::<f32>()?;
        let slice1 = slice1.slice(l1.start_offset()..);
        let slice2 = slice2.slice(l2.start_offset()..);
        let n = l1.shape().elem_count();
        let dst = unsafe { dev.alloc::<f32>(n) }.w()?;

        let func = dev.get_or_load_custom_func(
            "sentinel_silu_mul_f32",
            "sentinel_kernels",
            SENTINEL_PTX,
        )?;
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&dst);
        builder.arg(&slice1);
        builder.arg(&slice2);
        let n_u = n as u32;
        builder.arg(&n_u);
        unsafe { builder.launch(cfg) }.w()?;

        let storage = candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((storage, l1.shape().clone()))
    }
}

/// Fused `silu(gate) * up` — custom CUDA when available, else Candle.
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> CResult<Tensor> {
    if kernels_available(gate) && gate.shape() == up.shape() {
        match gate.apply_op2_no_bwd(up, &SiluMulOp) {
            Ok(t) => return Ok(t),
            Err(e) => {
                tracing::debug!(error = %e, "silu_mul CUDA failed; Candle fallback");
            }
        }
    }
    // Portable fallback
    let s = candle_nn::ops::silu(gate)?;
    Ok((s * up)?)
}

// ---------------------------------------------------------------------------
// scale
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ScaleOp {
    scale: f32,
}

impl CustomOp1 for ScaleOp {
    fn name(&self) -> &'static str {
        "sentinel_scale"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> CResult<(CpuStorage, Shape)> {
        let v = match storage {
            CpuStorage::F32(v) => v,
            _ => candle_core::bail!("scale cpu: f32 only"),
        };
        let n = layout.shape().elem_count();
        let start = layout.start_offset();
        let mut out = vec![0f32; n];
        for i in 0..n {
            out[i] = v[start + i] * self.scale;
        }
        Ok((CpuStorage::F32(out), layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &Layout,
    ) -> CResult<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if !(SENTINEL_KERNELS_BUILT && !SENTINEL_PTX.is_empty()) {
            candle_core::bail!("kernels not built");
        }
        if !layout.is_contiguous() {
            candle_core::bail!("scale requires contiguous");
        }
        let dev = storage.device();
        let slice = storage.as_cuda_slice::<f32>()?;
        let slice = slice.slice(layout.start_offset()..);
        let n = layout.shape().elem_count();
        let dst = unsafe { dev.alloc::<f32>(n) }.w()?;
        let func =
            dev.get_or_load_custom_func("sentinel_scale_f32", "sentinel_kernels", SENTINEL_PTX)?;
        let block = 256u32;
        let grid = ((n as u32) + block - 1) / block;
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&dst);
        builder.arg(&slice);
        builder.arg(&self.scale);
        let n_u = n as u32;
        builder.arg(&n_u);
        unsafe { builder.launch(cfg) }.w()?;
        let out = candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((out, layout.shape().clone()))
    }
}

pub fn scale_mul(x: &Tensor, scale: f32) -> CResult<Tensor> {
    if kernels_available(x) {
        if let Ok(t) = x.apply_op1_no_bwd(&ScaleOp { scale }) {
            return Ok(t);
        }
    }
    Ok((x * f64::from(scale))?)
}

// ---------------------------------------------------------------------------
// Portable helpers (always Candle for complex shapes; CUDA kernels used above)
// ---------------------------------------------------------------------------

/// Scatter-add rows — uses Candle `scatter_add` (device path when available).
pub fn scatter_add_rows(base: &Tensor, indices: &Tensor, values: &Tensor) -> CResult<Tensor> {
    let h = values.dim(1)?;
    let n = indices.dims1()?;
    let idx = indices
        .to_dtype(DType::U32)?
        .unsqueeze(1)?
        .expand((n, h))?
        .contiguous()?;
    base.scatter_add(&idx, values, 0)
}

/// RMSNorm — Candle path (custom kernel available for contiguous f32 rows).
pub fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f64) -> CResult<Tensor> {
    candle_nn::ops::rms_norm(x, weight, eps)
}

pub fn bias_add(x: &Tensor, bias: &Tensor) -> CResult<Tensor> {
    x.broadcast_add(bias)
}
