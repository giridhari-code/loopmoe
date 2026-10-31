# Sentinel v1.0

Looped MoE + Fertile + **Tesla** (DSpark) + DoBu kernels + **single/multi-GPU parallel** + recipe API + Rust training.

## Tesla (was DSpark)

Confidence-scheduled speculative decoding with **chunk verify**:

```bash
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" --tesla --tesla-k 5
```

| Stage | What happens |
|--------|----------------|
| **Draft** | Up to `k` tokens (greedy or sampled) |
| **Confidence** | GPU-friendly max-softmax + optional top-2 margin |
| **Schedule** | Survival product → how many to verify |
| **Chunk verify** | **One sequence forward** (big matmuls/attention once) — not N sequential steps |
| **Accept** | Match → keep; mismatch → target token + stop |
| **KV rewind** | Restore cache before rejected draft |

Stats: `drafted / scheduled / accepted / forwards / accept_rate`

Chunk verify = fewer forwards, better use of matrix multiplication bandwidth.

## Parallelism (single + multi GPU)

| Mode | How |
|------|-----|
| **Single GPU** (default) | `ParallelConfig::single()` |
| **Expert Parallel** | `ep_size > 1` (DeepEP-style stubs) |
| **Tensor / Data / Hybrid** | `tp_size`, `dp_size`, mesh ranks |

```bash
export WORLD_SIZE=8 RANK=0 LOCAL_RANK=0 EP_SIZE=8 TP_SIZE=1
```

## DoBu kernels

- **GEMM** — DeepGEMM-style (Nn / Nt / Tn, grouped MoE)
- **MLA** — FlashMLA-style prefill + KV-cache decode

## Recipe / server

```bash
cargo run --release --features server --bin sentinel-server -- --model ./weights
```

## Training

See TRAINING.md.
