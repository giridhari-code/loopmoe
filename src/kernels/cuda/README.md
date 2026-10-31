# Sentinel custom CUDA kernels

## Ops

| Kernel | f32 | f16 | bf16 |
|--------|-----|-----|------|
| silu_mul | yes | yes | yes |
| scale | yes | yes | yes |
| scatter_add_rows | yes | yes | yes |
| **tiled GEMM NN** (16×16) | yes | yes | yes |
| tiled GEMM NT | yes | — | — |

## Tiled GEMM

- Shared-memory **16×16** tiles, **float accumulate**
- DoBu `dobu_gemm_*` + grouped expert GEMM call `tiled_gemm_nn` first
- Auto-fallback to Candle/cuBLAS if PTX missing or size outside window

```bash
export CUDA_COMPUTE_CAP=89
cargo build --release
```
