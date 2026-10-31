//! Sentinel CUDA ops — f32 / f16 / bf16 silu_mul, scale, scatter_add_rows.

use candle_core::{
    CpuStorage, CustomOp1, CustomOp2, DType, Layout, Result as CResult, Shape, Tensor,
};

use super::{SENTINEL_KERNELS_BUILT, SENTINEL_PTX};

pub fn kernels_available(t: &Tensor) -> bool {
    SENTINEL_KERNELS_BUILT
        && t.device().is_cuda()
        && matches!(t.dtype(), DType::F32 | DType::F16 | DType::BF16)
}

fn ptx_ready() -> bool {
    SENTINEL_KERNELS_BUILT && !SENTINEL_PTX.is_empty()
}

fn silu_kernel_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("sentinel_silu_mul_f32"),
        DType::F16 => Some("sentinel_silu_mul_f16"),
        DType::BF16 => Some("sentinel_silu_mul_bf16"),
        _ => None,
    }
}

fn scale_kernel_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("sentinel_scale_f32"),
        DType::F16 => Some("sentinel_scale_f16"),
        DType::BF16 => Some("sentinel_scale_bf16"),
        _ => None,
    }
}

fn scatter_kernel_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("sentinel_scatter_add_rows_f32"),
        DType::F16 => Some("sentinel_scatter_add_rows_f16"),
        DType::BF16 => Some("sentinel_scatter_add_rows_bf16"),
        _ => None,
    }
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
        // Prefer f32 CPU path; other dtypes fall back via caller.
        let g = match s1 {
            CpuStorage::F32(v) => v,
            _ => candle_core::bail!("silu_mul cpu: use Candle fallback for non-f32"),
        };
        let u = match s2 {
            CpuStorage::F32(v) => v,
            _ => candle_core::bail!("silu_mul cpu: use Candle fallback for non-f32"),
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

        if !ptx_ready() {
            candle_core::bail!("sentinel CUDA kernels not built");
        }
        if !l1.is_contiguous() || !l2.is_contiguous() {
            candle_core::bail!("silu_mul requires contiguous tensors");
        }
        let dtype = s1.dtype();
        let kname = silu_kernel_name(dtype).ok_or_else(|| {
            candle_core::Error::Msg(format!("silu_mul: unsupported dtype {dtype:?}"))
        })?;
        let dev = s1.device();
        let n = l1.shape().elem_count();
        if n == 0 || n > u32::MAX as usize {
            candle_core::bail!("silu_mul: element count exceeds CUDA kernel limit");
        }
        let block = 256u32;
        let n_u32 = n as u32;
        let grid = n_u32 / block + u32::from(n_u32 % block != 0);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_custom_func(kname, "sentinel_kernels", SENTINEL_PTX)?;
        let n_u = n as u32;

        match dtype {
            DType::F32 => {
                let a = s1.as_cuda_slice::<f32>()?.slice(l1.start_offset()..);
                let b = s2.as_cuda_slice::<f32>()?.slice(l2.start_offset()..);
                let dst = unsafe { dev.alloc::<f32>(n) }.w()?;
                let mut builder = func.builder();
                builder.arg(&dst);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&n_u);
                unsafe { builder.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                    l1.shape().clone(),
                ))
            }
            DType::F16 => {
                let a = s1.as_cuda_slice::<half::f16>()?.slice(l1.start_offset()..);
                let b = s2.as_cuda_slice::<half::f16>()?.slice(l2.start_offset()..);
                let dst = unsafe { dev.alloc::<half::f16>(n) }.w()?;
                let mut builder = func.builder();
                builder.arg(&dst);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&n_u);
                unsafe { builder.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                    l1.shape().clone(),
                ))
            }
            DType::BF16 => {
                let a = s1.as_cuda_slice::<half::bf16>()?.slice(l1.start_offset()..);
                let b = s2.as_cuda_slice::<half::bf16>()?.slice(l2.start_offset()..);
                let dst = unsafe { dev.alloc::<half::bf16>(n) }.w()?;
                let mut builder = func.builder();
                builder.arg(&dst);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&n_u);
                unsafe { builder.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                    l1.shape().clone(),
                ))
            }
            _ => candle_core::bail!("silu_mul: dtype"),
        }
    }
}

pub fn silu_mul(gate: &Tensor, up: &Tensor) -> CResult<Tensor> {
    if kernels_available(gate) && gate.dtype() == up.dtype() && gate.shape() == up.shape() {
        if let Ok(t) = gate.apply_op2_no_bwd(up, &SiluMulOp) {
            return Ok(t);
        }
    }
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

        if !ptx_ready() {
            candle_core::bail!("kernels not built");
        }
        if !layout.is_contiguous() {
            candle_core::bail!("scale requires contiguous");
        }
        let dtype = storage.dtype();
        let kname = scale_kernel_name(dtype).ok_or_else(|| {
            candle_core::Error::Msg(format!("scale: unsupported {dtype:?}"))
        })?;
        let dev = storage.device();
        let n = layout.shape().elem_count();
        if n == 0 || n > u32::MAX as usize {
            candle_core::bail!("scale: element count exceeds CUDA kernel limit");
        }
        let block = 256u32;
        let n_u32 = n as u32;
        let grid = n_u32 / block + u32::from(n_u32 % block != 0);
        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let func = dev.get_or_load_custom_func(kname, "sentinel_kernels", SENTINEL_PTX)?;
        let n_u = n as u32;

        match dtype {
            DType::F32 => {
                let s = storage.as_cuda_slice::<f32>()?.slice(layout.start_offset()..);
                let dst = unsafe { dev.alloc::<f32>(n) }.w()?;
                let mut b = func.builder();
                b.arg(&dst);
                b.arg(&s);
                b.arg(&self.scale);
                b.arg(&n_u);
                unsafe { b.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                    layout.shape().clone(),
                ))
            }
            DType::F16 => {
                let s = storage.as_cuda_slice::<half::f16>()?.slice(layout.start_offset()..);
                let dst = unsafe { dev.alloc::<half::f16>(n) }.w()?;
                let mut b = func.builder();
                b.arg(&dst);
                b.arg(&s);
                b.arg(&self.scale);
                b.arg(&n_u);
                unsafe { b.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                    layout.shape().clone(),
                ))
            }
            DType::BF16 => {
                let s = storage.as_cuda_slice::<half::bf16>()?.slice(layout.start_offset()..);
                let dst = unsafe { dev.alloc::<half::bf16>(n) }.w()?;
                let mut b = func.builder();
                b.arg(&dst);
                b.arg(&s);
                b.arg(&self.scale);
                b.arg(&n_u);
                unsafe { b.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                    layout.shape().clone(),
                ))
            }
            _ => candle_core::bail!("scale dtype"),
        }
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
// scatter_add_rows — CustomOp2(base, values) + indices carried in struct
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ScatterAddOp {
    /// Device tensor of int32 indices, length = values.dim(0).
    /// Stored as raw pointer is unsafe across devices; we re-resolve from Tensor id
    /// by requiring caller to pass contiguous int32 via a side channel.
    /// Instead: indices are the first input through a different design —
    /// we pack nothing here and use a three-tensor path via storage launch helper.
    n: u32,
    h: u32,
}

/// Scatter-add using custom CUDA kernel when possible.
/// `indices`: 1-D int/uint token ids (length n), `values`: (n, h), `base`: (t, h).
pub fn scatter_add_rows(base: &Tensor, indices: &Tensor, values: &Tensor) -> CResult<Tensor> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(t) = scatter_add_rows_cuda(base, indices, values) {
            return Ok(t);
        }
    }
    // Candle fallback (works all dtypes / devices)
    let h = values.dim(1)?;
    let n = indices.dims1()?;
    let idx = indices
        .to_dtype(DType::U32)?
        .unsqueeze(1)?
        .expand((n, h))?
        .contiguous()?;
    base.scatter_add(&idx, values, 0)
}

#[cfg(feature = "cuda")]
fn scatter_add_rows_cuda(base: &Tensor, indices: &Tensor, values: &Tensor) -> CResult<Tensor> {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::WrapErr;
    use candle_core::backend::BackendDevice;

    if !ptx_ready() || !base.device().is_cuda() {
        candle_core::bail!("no cuda kernels");
    }
    let dtype = base.dtype();
    let kname = scatter_kernel_name(dtype).ok_or_else(|| {
        candle_core::Error::Msg("scatter dtype".into())
    })?;
    if values.dtype() != dtype {
        candle_core::bail!("dtype mismatch");
    }

    let (t, h) = base.dims2()?;
    let n = indices.dims1()?;
    if n == 0 || h == 0 || t == 0 {
        candle_core::bail!("scatter dimensions must be non-zero");
    }
    if n > u32::MAX as usize || h > u32::MAX as usize || t > u32::MAX as usize {
        candle_core::bail!("scatter dimension exceeds CUDA kernel limit");
    }
    if values.dims() != [n, h] {
        candle_core::bail!("values shape must be (n, h)");
    }

    // Clone base into mutable output (scatter-add accumulates into a copy)
    let out = base.contiguous()?;
    let idx_i32 = indices.to_dtype(DType::I32)?.contiguous()?;
    let vals = values.contiguous()?;

    struct ScatterPair {
        idx: Tensor,
        n: u32,
        h: u32,
        out_rows: u32,
    }
    impl CustomOp2 for ScatterPair {
        fn name(&self) -> &'static str {
            "sentinel_scatter_add_rows"
        }
        fn cpu_fwd(
            &self,
            _s1: &CpuStorage,
            _l1: &Layout,
            _s2: &CpuStorage,
            _l2: &Layout,
        ) -> CResult<(CpuStorage, Shape)> {
            candle_core::bail!("scatter custom op is CUDA-only")
        }
        fn cuda_fwd(
            &self,
            s_out: &candle_core::CudaStorage,
            l_out: &Layout,
            s_val: &candle_core::CudaStorage,
            l_val: &Layout,
        ) -> CResult<(candle_core::CudaStorage, Shape)> {
            if !ptx_ready() {
                candle_core::bail!("no ptx");
            }
            let dtype = s_out.dtype();
            let kname = scatter_kernel_name(dtype).ok_or_else(|| {
                candle_core::Error::Msg("dtype".into())
            })?;
            let dev = s_out.device();
            let func = dev.get_or_load_custom_func(kname, "sentinel_kernels", SENTINEL_PTX)?;

            // indices tensor storage
            let idx_storage = match self.idx.storage_and_layout() {
                (candle_core::Storage::Cuda(s), layout) => (s, layout),
                _ => candle_core::bail!("indices must be cuda"),
            };
            let idx_slice = idx_storage
                .0
                .as_cuda_slice::<i32>()?
                .slice(idx_storage.1.start_offset()..);

            // Work in-place conceptually: alloc dst = copy of out, atomic add values
            let n_elem = l_out.shape().elem_count();
            let cfg = LaunchConfig {
                grid_dim: (self.n, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };

            match dtype {
                DType::F32 => {
                    let src = s_out.as_cuda_slice::<f32>()?.slice(l_out.start_offset()..);
                    let dst = unsafe { dev.alloc::<f32>(n_elem) }.w()?;
                    // copy base → dst
                    dev.dtod_copy(&src.slice(0..n_elem), &dst).w()?;
                    let val = s_val.as_cuda_slice::<f32>()?.slice(l_val.start_offset()..);
                    let mut b = func.builder();
                    b.arg(&dst);
                    b.arg(&idx_slice);
                    b.arg(&val);
                    b.arg(&self.n);
                    b.arg(&self.h);
                    b.arg(&self.out_rows);
                    unsafe { b.launch(cfg) }.w()?;
                    Ok((
                        candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                        l_out.shape().clone(),
                    ))
                }
                DType::F16 => {
                    let src = s_out.as_cuda_slice::<half::f16>()?.slice(l_out.start_offset()..);
                    let dst = unsafe { dev.alloc::<half::f16>(n_elem) }.w()?;
                    dev.dtod_copy(&src.slice(0..n_elem), &dst).w()?;
                    let val = s_val.as_cuda_slice::<half::f16>()?.slice(l_val.start_offset()..);
                    let mut b = func.builder();
                    b.arg(&dst);
                    b.arg(&idx_slice);
                    b.arg(&val);
                    b.arg(&self.n);
                    b.arg(&self.h);
                    b.arg(&self.out_rows);
                    unsafe { b.launch(cfg) }.w()?;
                    Ok((
                        candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                        l_out.shape().clone(),
                    ))
                }
                DType::BF16 => {
                    let src = s_out.as_cuda_slice::<half::bf16>()?.slice(l_out.start_offset()..);
                    let dst = unsafe { dev.alloc::<half::bf16>(n_elem) }.w()?;
                    dev.dtod_copy(&src.slice(0..n_elem), &dst).w()?;
                    let val = s_val.as_cuda_slice::<half::bf16>()?.slice(l_val.start_offset()..);
                    let mut b = func.builder();
                    b.arg(&dst);
                    b.arg(&idx_slice);
                    b.arg(&val);
                    b.arg(&self.n);
                    b.arg(&self.h);
                    b.arg(&self.out_rows);
                    unsafe { b.launch(cfg) }.w()?;
                    Ok((
                        candle_core::CudaStorage::wrap_cuda_slice(dst, dev.clone()),
                        l_out.shape().clone(),
                    ))
                }
                _ => candle_core::bail!("scatter dtype"),
            }
        }
    }

    let op = ScatterPair {
        idx: idx_i32,
        n: n as u32,
        h: h as u32,
        out_rows: t as u32,
    };
    out.apply_op2_no_bwd(&vals, &op)
}

pub fn rmsnorm(x: &Tensor, weight: &Tensor, eps: f64) -> CResult<Tensor> {
    candle_nn::ops::rms_norm(x, weight, eps)
}

pub fn bias_add(x: &Tensor, bias: &Tensor) -> CResult<Tensor> {
    x.broadcast_add(bias)
}
