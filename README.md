# loopmoe

High-performance MoE + Looped Transformer inference framework in pure Rust.  
GPU-first (CUDA) with CPU fallback. Safetensors support.

> **Professional structure** (inspired by production inference engines)

```
loopmoe/
├── Cargo.toml
├── src/
│   ├── main.rs                    # server bootstrap
│   ├── lib.rs
│   ├── config.rs                  # ModelConfig, ServerConfig
│   ├── error.rs                   # unified Error enum
│   │
│   ├── server/
│   │   ├── mod.rs
│   │   ├── routes.rs              # /v1/completions, /v1/chat/completions
│   │   ├── openai_types.rs        # request/response structs
│   │   ├── sse.rs                 # streaming
│   │   └── middleware.rs          # auth, rate-limit, logging
│   │
│   ├── scheduler/
│   │   ├── mod.rs
│   │   ├── continuous.rs          # continuous batching scheduler
│   │   ├── sequence.rs            # Sequence { id, tokens, state, ... }
│   │   └── block_manager.rs       # paged KV-cache block allocator
│   │
│   ├── engine/
│   │   ├── mod.rs
│   │   ├── executor.rs            # runs forward pass on batch
│   │   ├── sampler.rs             # top-k, top-p, temperature, repetition penalty
│   │   ├── stop_criteria.rs       # EOS, max_tokens, stop strings
│   │   └── metrics.rs             # tokens/sec, latency, queue depth
│   │
│   ├── model/
│   │   ├── mod.rs
│   │   ├── loader.rs              # safetensors → weights
│   │   ├── llama.rs               # Llama architecture
│   │   ├── qwen.rs                # Qwen architecture
│   │   ├── attention.rs           # PagedAttention kernel launcher
│   │   ├── mlp.rs                 # SwiGLU
│   │   ├── rope.rs                # rotary embeddings
│   │   └── quantization.rs        # FP8/INT8/AWQ loaders
│   │
│   ├── kernels/                   # CUDA kernels (Rust bindings)
│   │   ├── mod.rs
│   │   ├── paged_attention.rs
│   │   ├── rmsnorm.rs
│   │   ├── rope.rs
│   │   ├── silu_mul.rs
│   │   ├── gemm_fp8.rs
│   │   └── cuda/
│   │       ├── mod.rs
│   │       ├── build.rs
│   │       ├── paged_attention.cu
│   │       ├── rmsnorm.cu
│   │       ├── rope.cu
│   │       ├── silu_mul.cu
│   │       └── gemm_fp8.cu
│   │
│   └── utils/
│       ├── mod.rs
│       ├── tokenizer.rs
│       └── logger.rs
│
├── tests/
│   ├── integration_server.rs
│   ├── test_scheduler.rs
│   └── test_kv_cache.rs
│
├── benches/
│   └── throughput.rs
│
├── examples/
│   ├── simple_completion.rs
│   └── chat_stream.rs
│
├── Dockerfile
├── README.md
└── config/
    └── default.toml
```

## Status

- **Scaffold complete** on branch `professional-structure`
- Existing specialized code (Looped-MoE, Tesla, Fertile, DoBu, training, recipe) is still present and will be gradually migrated into the new modules.
- New modules are currently placeholders ready for implementation.

## Quick start (legacy)

```bash
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" --tesla
```

## Server (when wired)

```bash
cargo run --release --bin sentinel-server --features server
```

## License

Apache-2.0
