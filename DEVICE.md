# Device policy (GPU first)

**Rule:** Always try **GPU (CUDA) first**. Use **CPU only when no CUDA device is available**.

This is the opposite of “prefer CPU” or “force CPU for tests”. CPU is a **fallback**, not the default path.

---

## Meaning of fallback

| Order | Device | When |
|-------|--------|------|
| 1 | CUDA (`cuda:0` or `local_rank`) | Preferred — train, infer, tests |
| 2 | CPU | Only if CUDA init fails |

**Wrong (old / inverted):** require CPU, or treat GPU as optional side path.  
**Right:** try CUDA → on failure, log a clear warning → run on CPU.

---

## CUDA toolkit versions

| Toolkit | Support |
|---------|---------|
| 11.x / 12.x | Yes |
| **13.0+** (Colab current) | Yes — requires **candle-core/nn ≥ 0.9.2** (cudarc ≥ 0.17 / 0.19) |

`candle 0.8` + `cudarc 0.13` panics with:

```text
Unsupported cuda toolkit version: `13.0`
```

loopmoe is pinned to **candle 0.9.2** so Colab CUDA 13 builds work with the default `cuda` feature.

Optional env (usually not needed with 0.9.2):

```bash
export CUDA_COMPUTE_CAP=7.5   # T4
# export CUDA_HOME=/usr/local/cuda
```

---

## Implementation map

| Location | Behaviour |
|----------|-----------|
| `src/bin/sentinel_train.rs` → `select_device()` | `Device::new_cuda(0)` then warn + `Device::Cpu` |
| `src/parallel/mesh.rs` → `DeviceMesh::select_device` | `new_cuda(local_rank)` → `new_cuda(0)` → CPU + warning |
| Inference engine | Uses `DeviceMesh::select_device` |
| Tests | Must follow the same order: **CUDA first**, CPU only if CUDA missing; print a clear warning on fallback |

---

## Tests

- Do **not** write tests that *require* CPU or that fail when CUDA is present.
- Correct pattern:

```text
1. Try CUDA
2. If Ok → run assertions on GPU
3. If Err → eprintln! / tracing warn: "CUDA unavailable — running on CPU"
4. Continue on CPU (still pass if logic is device-agnostic)
```

Strict “CUDA-only or fail” is allowed only for explicit `#[ignore]` / `cuda` feature-gated tests, not for default CI on machines without a GPU.

---

## Colab / operators

1. Runtime type: **GPU** (T4 / A100).
2. `git pull` then build with default `cuda` feature (candle 0.9.2).
3. `--no-default-features` (CPU-only build) only when there is no GPU or link still fails.

```bash
cargo build --release --bin sentinel-train
```

---

## Log line (canonical)

```text
CUDA unavailable — falling back to CPU
```

Train binary uses a similar message: `CUDA unavailable — training on CPU`.
