# Sentinel v1.0

Looped MoE + Fertile + **Tesla** (DSpark) + DoBu kernels + **single/multi-GPU parallel** + recipe API + Rust training.

## Parallelism (single + multi GPU)

| Mode | How |
|------|-----|
| **Single GPU** (default) | `ParallelConfig::single()` — full experts on one device |
| **Expert Parallel** | `ep_size > 1` — experts sharded (DeepEP-style dispatch/combine stubs) |
| **Tensor Parallel** | `tp_size > 1` — mesh ranks ready |
| **Data Parallel** | `dp_size > 1` |
| **Hybrid** | `tp × ep × dp == world_size` |

Env (optional):

```bash
export WORLD_SIZE=8 RANK=0 LOCAL_RANK=0 EP_SIZE=8 TP_SIZE=1
```

Collectives are currently **identity stubs** so single-process runs stay correct. Wire DeepEP / NCCL behind `AllToAllStub` for real multi-GPU.

## Tesla (was DSpark)

Confidence-scheduled speculative decoding:

```bash
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" --tesla --tesla-k 5
```

- Draft up to `k` tokens
- GPU-friendly confidence + optional top-2 margin
- Survival-product verify length + KV rewind

## DoBu kernels

- **GEMM** — DeepGEMM-style interface (Nn / Nt / Tn, grouped MoE GEMM)
- **MLA** — FlashMLA-style prefill + KV-cache decode

## Recipe / server

```bash
cargo run --release --features server --bin sentinel-server -- --model ./weights
```

## Training

See TRAINING.md.
