# LoopMoE

Inference framework for **Looped Transformer + Shared/Routed MoE**.

Kernel: **Fertile** (`src/kernels/fertile`)

## Pipeline

```
input_ids
  → embed
  → loop × loop_count:
        RMSNorm → Attention(RoPE, KV-cache) → residual
        RMSNorm → Fertile MoE (shared + top-k routed) → residual
  → RMSNorm → lm_head → logits
  → generate: prefill → decode (cache)
```

## Fertile MoE

```
y = Shared(x) + scale * Σ top-k routed experts
```

- Shared experts: always on, one pass
- Routed: top-k → expert-grouped buckets → **one run per expert**
- Name: `kernels::KERNEL_NAME == "Fertile"`

## Defaults

| | |
|--|--|
| loop_count | 24 |
| n_routed_experts | 64 |
| n_shared_experts | 2 |
| num_experts_per_tok | 6 |
| weights | safetensors |
| device | CUDA → CPU fallback |

## Status

- [x] Fertile MoE kernel path
- [x] RoPE + KV cache generation
- [x] Shared + routed experts
- [ ] Fertile CUDA backend
- [ ] Tokenizer / sampling / server

## Build

```bash
cargo build --release
cargo run --release -- --model /path/to/model.safetensors
```

Checkpoint experts: `0..n_routed-1` routed, then shared.
