//! FP8 E4M3 quantize / dequant / weight-only GEMM.

use candle_core::{DType, Device, Result as CResult, Tensor};

use super::{SENTINEL_FP8_BUILT, SENTINEL_FP8_PTX, SENTINEL_KERNELS_BUILT};

fn fp8_ready() -> bool {
    SENTINEL_FP8_BUILT && !SENTINEL_FP8_PTX.is_empty()
}

/// Per-tensor absmax scale for FP8 E4M3 (max finite ≈ 448).
pub fn fp8_scale_from_amax(amax: f32) -> f32 {
    const FP8_MAX: f32 = 448.0;
    if amax <= 0.0 || !amax.is_finite() {
        1.0
    } else {
        amax / FP8_MAX
    }
}

/// CPU quantize f32/bf16/f16 tensor → U8 FP8 E4M3 + scale.
pub fn quantize_to_fp8_e4m3(x: &Tensor) -> CResult<(Tensor, f32)> {
    let x32 = x.to_dtype(DType::F32)?.flatten_all()?;
    let v = x32.to_vec1::<f32>()?;
    let amax = v
        .iter()
        .map(|t| t.abs())
        .fold(0.0f32, f32::max);
    let scale = fp8_scale_from_amax(amax);
    let inv = if scale > 0.0 { 1.0 / scale } else { 1.0 };
    let bytes: Vec<u8> = v.iter().map(|&f| float_to_fp8_e4m3_host(f * inv)).collect();
    let out = Tensor::from_vec(bytes, x.shape(), x.device())?;
    Ok((out, scale))
}

/// CPU dequant U8 FP8 → f32.
pub fn dequant_fp8_e4m3(fp8: &Tensor, scale: f32) -> CResult<Tensor> {
    let flat = fp8.flatten_all()?;
    let bytes = flat.to_vec1::<u8>()?;
    let v: Vec<f32> = bytes
        .iter()
        .map(|&b| fp8_e4m3_to_float_host(b) * scale)
        .collect();
    Tensor::from_vec(v, fp8.shape(), fp8.device())?.to_dtype(DType::F32)
}

/// Weight-only GEMM: activations @ fp8_weights * scale.
/// `a`: (M, K) f32/bf16, `b_fp8`: (K, N) U8, `scale`: per-tensor.
/// Output dtype matches `a` when possible.
pub fn gemm_nn_fp8_weight(a: &Tensor, b_fp8: &Tensor, scale: f32) -> CResult<Tensor> {
    let (m, k1) = a.dims2()?;
    let (k2, n) = b_fp8.dims2()?;
    if m == 0 || n == 0 || k1 == 0 {
        candle_core::bail!("fp8 gemm dimensions must be non-zero");
    }
    if k1 != k2 {
        candle_core::bail!("fp8 gemm K mismatch");
    }
    if m > i32::MAX as usize || n > i32::MAX as usize || k1 > i32::MAX as usize {
        candle_core::bail!("fp8 gemm dimension exceeds CUDA kernel limit");
    }

    // Portable path: dequant weights → matmul (always correct)
    let fallback = || -> CResult<Tensor> {
        let w = dequant_fp8_e4m3(b_fp8, scale)?.to_dtype(a.dtype())?;
        a.matmul(&w)
    };

    if !a.device().is_cuda() || !fp8_ready() {
        return fallback();
    }

    // Prefer CUDA kernel via dequant+tiled if launch helpers fail
    // For robustness, use dequant on GPU through Candle after host scale
    // then tiled_gemm — still saves weight memory bandwidth if weights stay fp8 on disk.
    // True device FP8 GEMM when PTX loads:
    #[cfg(feature = "cuda")]
    {
        if let Ok(t) = gemm_nn_fp8_weight_cuda(a, b_fp8, scale, m, n, k1) {
            return Ok(t);
        }
    }
    fallback()
}

#[cfg(feature = "cuda")]
fn gemm_nn_fp8_weight_cuda(
    a: &Tensor,
    b_fp8: &Tensor,
    scale: f32,
    m: usize,
    n: usize,
    k: usize,
) -> CResult<Tensor> {
    use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
    use candle_core::cuda_backend::WrapErr;
    use candle_core::{CustomOp2, CpuStorage, Layout, Shape};

    // Dequant on GPU via kernel then standard matmul is simpler and stable.
    // Full in-kernel GEMM:
    let a_c = a.contiguous()?;
    let b_c = b_fp8.contiguous()?;

    #[derive(Clone)]
    struct Fp8Gemm {
        scale: f32,
        m: i32,
        n: i32,
        k: i32,
        output_dtype: DType,
    }
    impl CustomOp2 for Fp8Gemm {
        fn name(&self) -> &'static str {
            "sentinel_fp8_gemm"
        }
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> CResult<(CpuStorage, Shape)> {
            candle_core::bail!("cuda only")
        }
        fn cuda_fwd(
            &self,
            sa: &candle_core::CudaStorage,
            la: &Layout,
            sb: &candle_core::CudaStorage,
            lb: &Layout,
        ) -> CResult<(candle_core::CudaStorage, Shape)> {
            if !fp8_ready() {
                candle_core::bail!("fp8 ptx missing");
            }
            let dev = sa.device();
            let kname = match self.output_dtype {
                DType::F32 => "sentinel_gemm_nn_fp8w_f32",
                DType::F16 => "sentinel_gemm_nn_fp8w_f16",
                DType::BF16 => "sentinel_gemm_nn_fp8w_bf16",
                _ => candle_core::bail!("unsupported fp8 gemm dtype"),
            };
            let func =
                dev.get_or_load_custom_func(kname, "sentinel_fp8_kernels", SENTINEL_FP8_PTX)?;
            let tile = 16u32;
            let n_u32 = self.n as u32;
            let m_u32 = self.m as u32;
            let cfg = LaunchConfig {
                grid_dim: (
                    n_u32 / tile + u32::from(n_u32 % tile != 0),
                    m_u32 / tile + u32::from(m_u32 % tile != 0),
                    1,
                ),
                block_dim: (tile, tile, 1),
                shared_mem_bytes: 0,
            };
            let out_shape = Shape::from((self.m as usize, self.n as usize));
            let n_out = out_shape.elem_count();
            let b = sb.as_cuda_slice::<u8>()?.slice(lb.start_offset()..);

            match self.output_dtype {
                DType::F16 => {
                    let a = sa.as_cuda_slice::<half::f16>()?.slice(la.start_offset()..);
                    let c = unsafe { dev.alloc::<half::f16>(n_out) }.w()?;
                    let mut builder = func.builder();
                    builder.arg(&c);
                    builder.arg(&a);
                    builder.arg(&b);
                    builder.arg(&self.scale);
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
                    let c = unsafe { dev.alloc::<half::bf16>(n_out) }.w()?;
                let mut builder = func.builder();
                builder.arg(&c);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&self.scale);
                builder.arg(&self.m);
                builder.arg(&self.n);
                builder.arg(&self.k);
                unsafe { builder.launch(cfg) }.w()?;
                Ok((
                    candle_core::CudaStorage::wrap_cuda_slice(c, dev.clone()),
                    out_shape,
                ))
                }
                DType::F32 => {
                    let a = sa.as_cuda_slice::<f32>()?.slice(la.start_offset()..);
                let c = unsafe { dev.alloc::<f32>(n_out) }.w()?;
                let mut builder = func.builder();
                builder.arg(&c);
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&self.scale);
                builder.arg(&self.m);
                builder.arg(&self.n);
                builder.arg(&self.k);
                unsafe { builder.launch(cfg) }.w()?;
                    Ok((
                        candle_core::CudaStorage::wrap_cuda_slice(c, dev.clone()),
                        out_shape,
                    ))
                }
                _ => candle_core::bail!("unsupported fp8 gemm dtype"),
            }
        }
    }

    let output_dtype = a.dtype();
    let a_in = match output_dtype {
        DType::F32 | DType::F16 | DType::BF16 => a_c.to_dtype(output_dtype)?,
        _ => candle_core::bail!("unsupported fp8 gemm activation dtype"),
    };
    let op = Fp8Gemm {
        scale,
        m: m as i32,
        n: n as i32,
        k: k as i32,
        output_dtype,
    };
    a_in.apply_op2_no_bwd(&b_c, &op)
}

// ---- host FP8 codecs (match device) ----

fn fp8_e4m3_to_float_host(v: u8) -> f32 {
    let sign = (v >> 7) & 1;
    let exp = (v >> 3) & 0xF;
    let mant = v & 0x7;
    let f = if exp == 0 {
        if mant == 0 {
            0.0
        } else {
            // ldexp(mant, -9)
            (mant as f32) * (1.0 / 512.0)
        }
    } else if exp == 0xF {
        f32::NAN
    } else {
        (1.0 + (mant as f32) * 0.125) * 2f32.powi(exp as i32 - 7)
    };
    if sign == 1 {
        -f
    } else {
        f
    }
}

fn float_to_fp8_e4m3_host(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign = if x < 0.0 { 1u8 } else { 0u8 };
    let mut x = x.abs();
    if x == 0.0 {
        return sign << 7;
    }
    if x > 448.0 {
        x = 448.0;
    }
    // frexp
    let exp_f = x.log2().floor();
    let mut exp = exp_f as i32;
    let mut frac = x / 2f32.powi(exp); // [1, 2) roughly
    if frac < 1.0 {
        frac *= 2.0;
        exp -= 1;
    }
    let mut e = exp + 7;
    if e <= 0 {
        let den = x * 512.0;
        let m = den.round().clamp(0.0, 7.0) as u8;
        return (sign << 7) | m;
    }
    if e >= 15 {
        return (sign << 7) | 0x7E;
    }
    let mut m = ((frac - 1.0) * 8.0).round() as i32;
    if m > 7 {
        m = 0;
        e += 1;
        if e >= 15 {
            return (sign << 7) | 0x7E;
        }
    }
    (sign << 7) | ((e as u8 & 0xF) << 3) | (m as u8 & 7)
}

pub fn fp8_supported() -> bool {
    fp8_ready()
}
