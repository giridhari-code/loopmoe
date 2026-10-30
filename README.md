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
- [x] **Fused MoE kernel path** (expert-wise group, 1 run/expert)
- [ ] Efficient grouped GEMM / DeepEP-style CUDA kernels
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

**Fused kernel path** (`MoeLayer::forward`):
1. Shared experts — one pass
2. Router + top-k
3. Group all tokens by expert (single pass)
4. Each routed expert runs **exactly once**
5. Scaled scatter-add + combine

Checkpoint layout:
- `experts.0 .. experts.{n_routed-1}` → routed
- `experts.{n_routed} ..` → shared

## Build

```bash
cargo build --release
```

## Run

```bash
cargo run --release -- --model /path/to/model.safetensors
```
