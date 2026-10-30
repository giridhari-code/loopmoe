# LoopMoE

**MoE + Looped Transformer** inference framework written in pure Rust.

- GPU first (CUDA) with automatic CPU fallback
- Weights: `safetensors`
- Default loop count: **24**
- MoE follows **DeepSeek / DeepInfer** conventions (shared + routed experts)

## Status

- [x] Project structure
- [x] ModelConfig (loop_count = 24)
- [x] LoopedMoE forward skeleton
- [x] Device selection (CUDA → CPU)
- [x] Safetensors loading via candle
- [x] Top-k MoE routing
- [x] **Shared + Routed experts** (DeepSeek style)
- [ ] Efficient grouped GEMM / DeepEP-style kernels
- [ ] RoPE + KV cache
- [ ] Continuous batching / PagedAttention
- [ ] Tokenizer integration
- [ ] Sampling (temperature, top-p)
- [ ] OpenAI-compatible server

## MoE design (DeepInfer-inspired)

```
output = shared_experts(x) + routed_scaling_factor * Σ (w_k * routed_expert_k(x))
```

| Field | Default | Meaning |
|-------|---------|--------|
| `n_routed_experts` | 64 | Experts selected by router |
| `n_shared_experts` | 2 | Always-on experts |
| `num_experts_per_tok` | 6 | Top-k |
| `routed_scaling_factor` | 1.0 | Scale on routed path |
| `scoring_func` | softmax | or `sigmoid` |
| `norm_topk_prob` | true | Renormalize top-k weights |
| `topk_method` | greedy | Phase-1 |

Checkpoint layout expected:
- `experts.0 .. experts.{n_routed-1}` → routed
- `experts.{n_routed} ..` → shared

## Build

```bash
# CUDA (default)
cargo build --release

# CPU only
cargo build --release --no-default-features
```

## Run

```bash
cargo run --release -- --model /path/to/model.safetensors
```

## Design Notes

- Physical layers (`num_layers`) are weight-tied and applied `loop_count` times.
- Effective depth = `num_layers * loop_count` (default 8 × 24 = 192).
- Future kernels will take inspiration from DeepInfer (DeepEP / DeepGEMM style dispatch).
