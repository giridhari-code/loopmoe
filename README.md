# LoopMoE

Looped Transformer + Shared/Routed MoE. Kernel: **Fertile**.

## Backends

| Condition | Backend |
|-----------|--------|
| CUDA device | `fertile-cuda` |
| aarch64, no GPU | `fertile-arm` (customer) |
| else | `fertile-cpu` |

```bash
cargo build --release                          # CUDA default
cargo build --release --no-default-features --features arm
cargo build --release --no-default-features --features cpu
```

## Pipeline

```
embed → [loop: Attn(RoPE,KV) → Fertile MoE] → lm_head → prefill/decode
```
