# LoopMoE

**MoE + Looped Transformer** inference framework written in pure Rust.

- GPU first (CUDA) with automatic CPU fallback
- Weights: `safetensors`
- Default loop count: **24**
- Architecture inspired by modern looped MoE research + DeepInfer scheduling ideas

## Status (Phase 1 Scaffold)

- [x] Project structure
- [x] ModelConfig (loop_count = 24)
- [x] LoopedMoE forward skeleton
- [x] MoE layer (top-k routing)
- [x] Device selection (CUDA → CPU)
- [x] Safetensors loading via candle
- [x] Proper top-k MoE routing (Phase 1)
- [ ] Efficient grouped GEMM / DeepEP-style kernels
- [ ] RoPE + KV cache
- [ ] Continuous batching / PagedAttention
- [ ] Tokenizer integration
- [ ] Sampling (temperature, top-p)
- [ ] OpenAI-compatible server

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
- MoE experts live inside each TransformerBlock.
- Future kernels will take inspiration from DeepInfer (FlashMLA, DeepGEMM, DeepEP style dispatch).
