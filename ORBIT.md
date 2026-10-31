# Orbit architecture

Orbit is **loopmoe’s native** continuous-batch inference runtime.

## Phases

| Phase | Status |
|-------|--------|
| O1 BlockLedger | Done |
| O2 / O2.1 PackedBatch B>1 | Done |
| O3 Tesla-in-schedule + chunked prefill | Done |
| **O4 Frontier server** | **Done** |
| O5 Prefix hash + EP | Planned |

## O4 Frontier

OpenAI-compatible HTTP API on top of `InferenceEngine` (+ optional Tesla).

```bash
cargo run --release --bin sentinel-server --features server -- \
  --model ./weights --port 8080 --tesla --tesla-k 5
```

| Route | Method |
|-------|--------|
| `/health` | GET |
| `/v1/models` | GET |
| `/v1/chat/completions` | POST (JSON or `stream: true` SSE) |
| `/v1/completions` | POST |

Module: `src/frontier/` · binary: `sentinel-server`.

Generate path today: mutex + `generate` / `generate_tesla`. Continuous `BatchEngine` worker can replace `generate()` later without changing routes.
