# Sentinel custom CUDA kernels

## Source

`sentinel.cu` — real device kernels:

| Kernel | Purpose |
|--------|---------|
| `sentinel_silu_mul_f32` | Fused SwiGLU gate: `silu(gate) * up` |
| `sentinel_scatter_add_rows_f32` | MoE token scatter-add |
| `sentinel_rmsnorm_f32` | RMSNorm |
| `sentinel_scale_f32` | Attention scale |
| `sentinel_bias_add_f32` | Bias broadcast |

## Build

Requires **nvcc** on `PATH` and `--features cuda` (default):

```bash
# optional: target your GPU
export CUDA_COMPUTE_CAP=89   # e.g. Ada 8.9

cargo build --release
# look for: "Sentinel CUDA kernels built → .../sentinel_kernels.ptx"
```

Without nvcc the crate still builds; ops use Candle fallbacks.

## Usage

```rust
use loopmoe::kernels::cuda::{silu_mul, kernels_built};

// Expert FFN already calls silu_mul internally.
assert!(kernels_built()); // true if PTX was generated
```

## Notes

- First kernels are **f32**. BF16/FP16 variants are next (same `.cu` with `__half` / `__nv_bfloat16`).
- Full GEMM stays on cuBLAS via Candle `matmul` — custom tiled GEMM is a later milestone.
