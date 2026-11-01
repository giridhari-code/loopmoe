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
2. Build with default `cuda` feature when a GPU is available.
3. `--no-default-features` (CPU-only build) only when there is no GPU or CUDA toolkit link fails.

---

## Log line (canonical)

```text
CUDA unavailable — falling back to CPU
```

Train binary uses a similar message: `CUDA unavailable — training on CPU`.
