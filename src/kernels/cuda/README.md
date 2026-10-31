# Sentinel custom CUDA kernels

## Kernels (`sentinel.cu`)

| Kernel | f32 | f16 | bf16 |
|--------|-----|-----|------|
| `sentinel_silu_mul_*` | yes | yes | yes |
| `sentinel_scale_*` | yes | yes | yes |
| `sentinel_scatter_add_rows_*` | yes | yes | yes |
| `sentinel_rmsnorm_f32` | yes | — | — |

## Build

```bash
export CUDA_COMPUTE_CAP=89   # optional
cargo build --release --features cuda
```

Needs `nvcc`. Without it → Candle fallbacks still work.

## Wiring

- **Expert FFN** → `silu_mul` (BF16 path used by default model dtype)
- **Fertile** `device_scatter_add` → `scatter_add_rows` custom kernel

## Notes

- f16/bf16 atomics use hardware `atomicAdd` on sm_70+ / sm_80+, else CAS loop.
- Full GEMM remains cuBLAS via Candle.
