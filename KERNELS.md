# Kernels architecture (clean start)

## Rule

| Layer | Language | Role |
|-------|----------|------|
| Host (engine, train, Orbit, dispatch) | **Rust** | Tensors, scheduling, autograd path |
| Device kernels (GPU) | **CUDA C++** (`.cu` / `.cuh`) | Actual FLOPs on NVIDIA GPUs |

Do **not** write GPU kernels in pure Rust. Optional research tools (Triton) are out of tree.

```text
Rust call site
  → candle Tensor / custom op
  → PTX loaded from build.rs (nvcc)
  → CUDA C++ __global__ kernels
```

## Layout

```text
src/kernels/
  mod.rs              # public API
  diff_ops.rs         # train: force Candle ops (no broken bwd)
  fertile/            # MoE routing (Rust + backend)
  dobu/               # GEMM / MLA dispatch (Rust)
  cuda/
    README.md
    sentinel.cu       # fused silu_mul, rmsnorm, residual, tiled gemm, …
    fp8_kernels.cu    # FP8 quant / dequant / gemm helpers
    fp8.cuh
    ops.rs            # Rust wrappers around PTX / Candle fallback
    gemm.rs
    fp8.rs
    mod.rs
  cpu/                # CPU fallback hooks
```

**Removed (old empty stubs):** orphan `.cu` placeholders that were never wired into `build.rs` and did nothing (`paged_attention.cu`, `rmsnorm.cu`, `rope.cu`, `silu_mul.cu`, `gemm_fp8.cu` as empty files). Real fused ops live in **`sentinel.cu`**.

## Build

```bash
# T4
export CUDA_COMPUTE_CAP=75    # or 7.5 — both accepted by build.rs
cargo build --release --features cuda

# A100
export CUDA_COMPUTE_CAP=80
```

`build.rs` runs `nvcc -ptx` on `sentinel.cu` + `fp8_kernels.cu`. If `nvcc` fails, build **continues**; Candle ops still work.

## Colab notes

| Issue | Action |
|-------|--------|
| Toolkit 13.0 | `candle` **0.9.2** (cudarc 0.19) |
| T4 + candle-kernels MoE WMMA | sm_75 cannot build those upstream kernels → use `--no-default-features` for **train** on free T4, or A100 runtime |
| Custom Sentinel PTX | Independent; sm_75-friendly kernels in `sentinel.cu` |

## Adding a new kernel

1. Implement `__global__` in `src/kernels/cuda/sentinel.cu` (or new `.cu` + register in `build.rs`).
2. Expose a Rust function in `ops.rs` / `gemm.rs` with Candle `Tensor` in/out and **CPU/Candle fallback**.
3. Call from `model/` or `fertile/` only through that Rust API.
4. Keep train path on `DiffOpsGuard` (differentiable Candle ops).

## Policy link

GPU first, CPU fallback: see **DEVICE.md**.
