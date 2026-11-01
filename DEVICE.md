# Device policy (GPU first)

**Rule:** Always try **GPU (CUDA) first**. Use **CPU only when no CUDA device is available**.

---

## CUDA toolkit + GPU arch

| Issue | Fix |
|-------|-----|
| Toolkit **13.0** (Colab) | `candle ≥ 0.9.2` (cudarc 0.19) — we use **0.10.2** |
| T4 **sm_75** + bf16 WMMA compile error | `candle ≥ 0.10` passes `-DNO_BF16_KERNEL` when compute_cap < 80 ([PR #3367](https://github.com/huggingface/candle/pull/3367)) |

```bash
export CUDA_COMPUTE_CAP=75   # T4 (integer form preferred)
cargo build --release --bin sentinel-train   # default features = cuda
```

---

## Fallback order

| Order | Device | When |
|-------|--------|------|
| 1 | CUDA | Preferred |
| 2 | CPU | Only if CUDA init fails |

Log: `CUDA unavailable — training on CPU` / `falling back to CPU`.

---

## Implementation

| Location | Behaviour |
|----------|-----------|
| `sentinel_train` `select_device()` | `new_cuda(0)` → warn + CPU |
| `DeviceMesh::select_device` | rank → cuda:0 → CPU |
| Tests | CUDA first, then CPU + warning |

---

## Colab GPU train

1. Runtime: **GPU**
2. `git pull origin main`
3. `CUDA_COMPUTE_CAP=75 cargo build --release --bin sentinel-train`
4. Train; log must show CUDA device, not CPU fallback
