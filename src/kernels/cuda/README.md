# Sentinel custom CUDA (device code)

**Language: CUDA C++ only** for `__global__` kernels. Rust wraps them — see repo root **KERNELS.md**.

## Built by `build.rs`

| File | Role |
|------|------|
| `sentinel.cu` | silu_mul, scale, scatter, residual, rmsnorm, tiled GEMM |
| `fp8_kernels.cu` | FP8 quant / dequant / weight GEMM helpers |
| `fp8.cuh` | shared headers |

## Rust wrappers

`ops.rs`, `gemm.rs`, `fp8.rs` — Candle `Tensor` API + fallback if PTX missing.

## Arch

```bash
export CUDA_COMPUTE_CAP=75   # T4
# export CUDA_COMPUTE_CAP=80 # A100
cargo build --release --features cuda
```
