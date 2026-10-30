# LoopMoE

Looped Transformer + Shared/Routed MoE. Kernel: **Fertile**.

## RAM optimizations

| Area | What |
|------|------|
| **KV cache** | Preallocated capacity (`max_seq_len`); no O(n²) `cat` growth |
| **RoPE** | Sin/cos table sized to `max_seq_len` (default **4096**), not 128k |
| **Prefill logits** | Only last-token → lm_head |
| **Decode mask** | No causal mask when `seq==1` |
| **Weights** | mmap safetensors + BF16 |
| **MoE** | Device-side group plan; one run per expert |

```rust
max_seq_len: 4096  // raise only if you need longer context
```

## Backends

CUDA → ARM (aarch64) → CPU

```bash
cargo build --release
```
