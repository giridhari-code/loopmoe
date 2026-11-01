# Kernels architecture

## Languages

| Layer | Language |
|-------|----------|
| Device kernels | **CUDA C++** (`.cu` / `.cuh`) |
| Host dispatch | **Rust** |

## Final folder map (`src/kernels/`)

```text
src/kernels/
├── mod.rs                 # public API
├── diff_ops.rs            # train: DiffOpsGuard
├── fertile/               # MoE routing
├── dobu/                  # GEMM / MLA dispatch
├── cpu/
└── cuda/
    ├── mod.rs
    ├── README.md
    ├── sentinel.cu        # fused silu, residual, rmsnorm, tiled GEMM, scatter
    ├── fp8_kernels.cu
    ├── fp8.cuh
    ├── ops.rs             # Rust launch + Candle fallback
    ├── gemm.rs
    └── fp8.rs
```

Root: `build.rs` runs `nvcc -ptx` on `sentinel.cu` + `fp8_kernels.cu`.

## Removed (stubs / unwired)

```text
cuda/paged_attention.cu, rmsnorm.cu, rope.cu, silu_mul.cu, gemm_fp8.cu, build.rs
paged_attention.rs, rmsnorm_paged.rs, rope_paged.rs, silu_mul_fused.rs, gemm_fp8.rs
```

## Arch

```bash
export CUDA_COMPUTE_CAP=75   # T4; also accepts 7.5
cargo build --release --features cuda
```

Colab T4: upstream `candle-kernels` MoE WMMA may still fail — use
`--no-default-features` for train binary. Custom `sentinel.cu` is sm_75-safe.

## Adding a kernel

1. Write `__global__` in `sentinel.cu` (or new `.cu` + `build.rs` entry).
2. Wrap in `ops.rs` / `gemm.rs` with Candle fallback.
3. Respect `force_diff_ops()` for train.
