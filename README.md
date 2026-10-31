# loopmoe

**High-performance MoE + Looped Transformer inference in pure Rust.**  
GPU-first (CUDA) · CPU fallback · Safetensors · OpenAI-compatible API

---

## Features

| Area | Capability |
|------|------------|
| **Architecture** | Looped MoE (Sentinel v1.0), MLA, GQA |
| **Kernels** | Fertile (MoE dispatch), DoBu (GEMM/MLA/FP8), custom CUDA |
| **Decoding** | Tesla speculative decoding (parallel tree + chain) |
| **Parallelism** | Expert Parallel / Tensor Parallel / Data Parallel mesh |
| **Serving** | OpenAI-compatible `/v1/chat/completions` (feature `server`) |
| **Training** | Pretrain · SFT · DPO · Mid-train pipelines |
| **Runtime** | Continuous-batching scheduler skeleton · Paged KV · Metrics |

---

## Quick Start

```bash
# Inference (Tesla speculative decoding)
cargo run --release --bin sentinel -- \
  --model ./weights \
  --prompt "Hello" \
  --tesla --tesla-k 4

# OpenAI-compatible server
cargo run --release --bin sentinel-server --features server -- \
  --model ./weights --port 8080

# Training
cargo run --release --bin sentinel-train -- --config configs/train_sft.json
```

---

## Project Layout

```
src/
├── main.rs / bin/          # CLI entrypoints
├── config.rs              # ModelConfig · ServerConfig
├── error.rs
│
├── engine/                # InferenceEngine + executor / sampler / metrics
├── model/                 # LoopedMoE · attention · MLA · MoE · RoPE
├── kernels/               # Fertile · DoBu · CUDA · CPU backends
├── scheduler/             # Continuous batching · Sequence · BlockManager
├── server/                # OpenAI routes · SSE · middleware  (feature = "server")
├── cache/                 # GQA / MLA KV-cache
├── tesla/                 # Speculative decoding (tree + verify)
├── parallel/              # Device mesh · EP / TP / DP
├── recipe/                # Chat templates · OpenAI types
├── train/                 # Pretrain / SFT / DPO
├── sampler.rs · tokenizer.rs
└── utils/
```

---

## Configuration

Default model config lives in `configs/sentinel-v1.0.json`.  
Server defaults in `config/default.toml`.

```toml
[server]
host = "0.0.0.0"
port = 8000
max_num_seqs = 256
max_num_batched_tokens = 8192
```

---

## Build Features

| Feature | Description |
|---------|-------------|
| `cuda` (default) | NVIDIA GPU path via Candle + custom kernels |
| `server` | Axum HTTP server + OpenAI endpoints |
| `cpu` / `arm` | CPU / ARM backends |

```bash
cargo build --release --features server,cuda
```

---

## Docker

```bash
docker build -t loopmoe .
docker run --gpus all -p 8000:8000 loopmoe
```

---

## License

Apache-2.0
