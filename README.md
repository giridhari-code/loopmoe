# loopmoe

**High-performance MoE + Looped Transformer inference in pure Rust.**  
GPU-first (CUDA) · CPU fallback · Safetensors · OpenAI-compatible API

Runtime architecture: **Orbit** (continuous batching + physical page ledger).

---

## Device policy

**GPU first.** CPU runs **only** when CUDA is unavailable (clear warning logged).  
Details: **[DEVICE.md](DEVICE.md)** — tests and train/infer must not invert this.

---

## Features

| Area | Capability |
|------|------------|
| **Architecture** | Looped MoE (Sentinel), MLA, GQA |
| **Orbit runtime** | Continuous batching · **BlockLedger** (paged KV/MLA addressing) · Tesla-aware schedule |
| **Kernels** | Fertile (MoE), DoBu (GEMM/MLA/FP8), fused RMSNorm / residual / silu_mul |
| **Decoding** | Tesla speculative (tree + MTP varlen verify) |
| **Parallelism** | Expert / Tensor / Data parallel mesh |
| **Serving** | OpenAI-compatible `/v1/chat/completions` (feature `server`) |
| **Training** | Pretrain · SFT · DPO pipelines |

---

## Orbit (runtime)

```
OrbitCore
  ├── OrbitSched    # who runs this step (decode / Tesla / prefill)
  ├── BlockLedger   # physical pages + slot_mapping   ← O1 done
  └── Pulse         # packed batch forward            ← O2 next
```

| Phase | Status |
|-------|--------|
| **O1** BlockLedger | **Done** — pages, block tables, slot_mapping |
| **O2** PackedBatch decode | Next |
| **O3** Chunked prefill + Tesla-in-schedule | Planned |
| **O4** Frontier server on Orbit | Planned |
| **O5** Prefix hash + EP collectives | Planned |

---

## Quick Start

```bash
# Inference (Tesla)
cargo run --release --bin sentinel -- \
  --model ./weights \
  --prompt "Hello" \
  --tesla --tesla-k 4

# Multi-request batch
cargo run --release --bin sentinel -- \
  --model ./weights \
  --prompts "Hi" "Explain MoE" \
  --max-tokens 64

# OpenAI-compatible server
cargo run --release --bin sentinel-server --features server -- \
  --model ./weights --port 8080
```

---

## Project Layout

```
src/
├── orbit/                 # Orbit runtime (BlockLedger, OrbitSched)
├── scheduler/             # Continuous queues (used by OrbitSched)
├── engine/                # InferenceEngine + BatchEngine
├── model/                 # LoopedMoE · attention · MLA · MoE
├── kernels/               # Fertile · DoBu · CUDA
├── tesla/                 # Speculative decoding
├── cache/                 # GQA / MLA tensors
├── parallel/              # Device mesh · EP plan
├── server/                # OpenAI routes (feature = server)
└── ...
```

---

## Build Features

| Feature | Description |
|---------|-------------|
| `cuda` (default) | NVIDIA GPU + custom kernels |
| `server` | Axum HTTP + OpenAI endpoints |
| `cpu` / `arm` | CPU / ARM backends |

```bash
cargo build --release --features server,cuda
```

---

## License

Apache-2.0
