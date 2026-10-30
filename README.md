# LoopMoE

**MoE + Looped Transformer** inference framework in pure Rust.

- GPU first (CUDA) + CPU fallback
- Safetensors weights
- loop_count default **24**
- DeepSeek-style shared + routed MoE
- **RoPE + KV cache** generation path

## Status

- [x] RoPE (rotary embeddings)
- [x] KV cache (prefill + decode)
- [x] Shared + Routed MoE + fused path
- [ ] DeepEP-style CUDA kernels
- [ ] PagedAttention / continuous batching
- [ ] Tokenizer + sampling
- [ ] OpenAI server

## Generation

```
1. Prefill(prompt) → LayerCaches
2. Decode(token)   → append cache, full-history attention
```

Looped model: cache written only on **last** loop iteration.

```bash
cargo build --release
cargo run --release -- --model /path/to/model.safetensors
```
