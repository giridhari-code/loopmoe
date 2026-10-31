//! Tiled GEMM via Sentinel CUDA kernels (shared-memory 16×16 tiles).
//! Falls back to Candle `matmul` (cuBLAS) when kernels unavailable.

use candle_core::{
    CustomOp2, DType, Layout, Result as CResult, Shape, Tensor, CpuStorage,
};

use super::{SENTINEL_KERNELS_BUILT, SENTINEL_PTX};

const TILE: u32 = 16;

fn ptx_ready() -> bool {
    SENTINEL_KERNELS_BUILT && !SENTINEL_PTX.is_empty()
}

fn gemm_nn_name(dtype: DType) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some("sentinel_gemm_nn_f32"),
        DType::F16 => Some("sentinel_gemm_nn_f16"),
        DType::BF16 => Some("sentinel_gemm_nn_bf16"),
        _ => None,
    }
}

/// Prefer tiled custom kernel for medium 2-D GEMMs on CUDA.
/// Very large GEMMs still often faster on cuBLAS — caller can force Candle.
pub fn should_use_tiled(m: usize, n: usize, k: usize) -> bool {
    // Use tiled for sizes where launch overhead is amortized but we stay
    // within a practical custom-kernel regime. Outside → cuBLAS.
    let elems = m.saturating_mul(n).saturating_mul(k);
    elems > 0 && elems <= 64 * 1024 * 1024 // ~64M MACs
}

#[derive(Debug, Clone, Copy)]
enum GemmKind {
    Nn,
    Nt,
}

#[derive(Debug, Clone)]
struct TiledGemmOp {
    kind: GemmKind,
    m: i32,
    n: i32,
    k: i32,
}

impl CustomOp2 for TiledGemmOp {
    fn name(&self) -> &'static str {
        "sentinel_tiled_gemm"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
    ) -> CResult<(CpuStorage, Shape)> {
        candle_core::bail!("tiled gemm CustomOp is CUDA-only")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        sa: &candle_core::CudaStorage,
        la: &Layout,
        sb: &candle_core::CudaStorage,
        lb: &Layout,
    ) -> CResult<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        if !ptx_ready() {
            candle_core::bail!("ptx missing");
        }
        if !la.is_contiguous() || !lb.is_contiguous() {
            candle_core::bail!("tiled gemm needs contiguous");
        }
        let dtype = sa.dtype();
        let kname = match self.kind {
            GemmKind::Nn => gemm_nn_name(dtype),
            GemmKind::Nt => {
                if dtype == DType::F32 {
                    Some("sentinel_gemm_nt_f32")
                } else {
                    None // NT only f32 for now; others via transpose + NN
                }
            }
        }
        .ok_or_else(|| candle_core::Error::Msg(format!("gemm dtype {dtype:?}")))?;

        let dev = sa.device();
        let func = dev.get_or_load_custom_func(kname, "sentinel_kernels", SENTINEL_PTX)?;

        let n_u32 = self.n as u32;
        let m_u32 = self.m as u32;
        let grid_x = n_u32 / TILE + u32::from(n_u32 % TILE != 0);
        let grid_y = m_u32 / TILE + u32::from(m_u32 % TILE != 0);
        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (TILE, TILE, 1),
            shared_mem_bytes: 0,
        };
        let out_shape = Shape::from((self.m as usize, self.n as usize));
        let n_out = out_shape.elem_count();

        match dtype {
            DType::F32 => {
                let a = sa.as_cuda_slice::<f32>()?.slice(la.start_offset()..);
                let b = sb.as_cuda_slice::<f32>()?.slice(lb.start_offset()..);
                let c = unsafe { dev.alloc::<f32>(n_out) }.w()?;
                let mut builder = func.builder();
                builder.arg(&c);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&self.m);
                builder.arg(&self.n);
                builder.arg(&self.k);
                unsafe { builder.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(c, dev.clone()),
                    out_shape,
                ))
            }
            DType::F16 => {
                let a = sa.as_cuda_slice::<half::f16>()?.slice(la.start_offset()..);
                let b = sb.as_cuda_slice::<half::f16>()?.slice(lb.start_offset()..);
                let c = unsafe { dev.alloc::<half::f16>(n_out) }.w()?;
                let mut builder = func.builder();
                builder.arg(&c);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&self.m);
                builder.arg(&self.n);
                builder.arg(&self.k);
                unsafe { builder.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(c, dev.clone()),
                    out_shape,
                ))
            }
            DType::BF16 => {
                let a = sa.as_cuda_slice::<half::bf16>()?.slice(la.start_offset()..);
                let b = sb.as_cuda_slice::<half::bf16>()?.slice(lb.start_offset()..);
                let c = unsafe { dev.alloc::<half::bf16>(n_out) }.w()?;
                let mut builder = func.builder();
                builder.arg(&c);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&self.m);
                builder.arg(&self.n);
                builder.arg(&self.k);
                unsafe { builder.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(c, dev.clone()),
                    out_shape,
                ))
            }
            _ => candle_core::bail!("gemm dtype"),
        }
    }
}

/// C = A @ B  with A:(M,K) B:(K,N)
pub fn tiled_gemm_nn(a: &Tensor, b: &Tensor) -> CResult<Tensor> {
    let (m, k1) = a.dims2()?;
    let (k2, n) = b.dims2()?;
    if k1 != k2 {
        candle_core::bail!("gemm_nn: K mismatch {k1} vs {k2}");
    }
    if m == 0 || n == 0 || k1 == 0 {
        candle_core::bail!("gemm_nn: dimensions must be non-zero");
    }
    if m > i32::MAX as usize || n > i32::MAX as usize || k1 > i32::MAX as usize {
        candle_core::bail!("gemm_nn: dimension exceeds CUDA kernel limit");
    }
    if a.device().is_cuda()
        && ptx_ready()
        && kernels_dtype_ok(a.dtype())
        && a.dtype() == b.dtype()
        && should_use_tiled(m, n, k1)
    {
        let a_c = a.contiguous()?;
        let b_c = b.contiguous()?;
        let op = TiledGemmOp {
            kind: GemmKind::Nn,
            m: m as i32,
            n: n as i32,
            k: k1 as i32,
        };
        if let Ok(t) = a_c.apply_op2_no_bwd(&b_c, &op) {
            return Ok(t);
        }
    }
    a.matmul(b)
}

/// C = A @ Bᵀ  with A:(M,K) B:(N,K)
pub fn tiled_gemm_nt(a: &Tensor, b: &Tensor) -> CResult<Tensor> {
    let (m, k1) = a.dims2()?;
    let (n, k2) = b.dims2()?;
    if k1 != k2 {
        candle_core::bail!("gemm_nt: K mismatch");
    }
    if m == 0 || n == 0 || k1 == 0 {
        candle_core::bail!("gemm_nt: dimensions must be non-zero");
    }
    if m > i32::MAX as usize || n > i32::MAX as usize || k1 > i32::MAX as usize {
        candle_core::bail!("gemm_nt: dimension exceeds CUDA kernel limit");
    }
    if a.device().is_cuda()
        && ptx_ready()
        && a.dtype() == DType::F32
        && b.dtype() == DType::F32
        && should_use_tiled(m, n, k1)
    {
        let a_c = a.contiguous()?;
        let b_c = b.contiguous()?;
        let op = TiledGemmOp {
            kind: GemmKind::Nt,
            m: m as i32,
            n: n as i32,
            k: k1 as i32,
        };
        if let Ok(t) = a_c.apply_op2_no_bwd(&b_c, &op) {
            return Ok(t);
        }
    }
    // Fallback: explicit transpose + NN (or cuBLAS)
    a.matmul(&b.t()?)
}

fn kernels_dtype_ok(d: DType) -> bool {
    matches!(d, DType::F32 | DType::F16 | DType::BF16)
}
