# DeepInfer → loopmoe mapping

[DeepInfer](https://github.com/deepseek-ai/DeepInfer) is DeepSeek's production inference stack
(Python + C++/CUDA). loopmoe is a pure-Rust cousin focused on Looped-MoE + Tesla.

This document records which DeepInfer ideas are already in loopmoe and what was ported.

## Feature map

| DeepInfer | loopmoe |
|-----------|---------|
| FlashMLA | `model/mla.rs` + `kernels/dobu/mla.rs` |
| DeepGEMM | `kernels/dobu/gemm.rs` + tiled CUDA GEMM |
| DeepEP (expert parallel) | `parallel/` mesh + EP plan (stubs for real collectives) |
| DSpark / MTP varlen | `tesla/` speculative decode (tree + confidence verify) |
| Continuous batch scheduler | `scheduler/continuous.rs` |
| SequenceStatus | `scheduler/sequence.rs` |
| BlockManager (ref-count free list) | `scheduler/block_manager.rs` |
| EPLB | future (`model_executor/eplb` patterns) |
| Async zero-overhead schedule | future |
| Batch-invariant / deterministic | future |
| Structured output (xgrammar) | future |
| OpenAI server | `server/` + `recipe/` |

## Status / finish reasons (DeepInfer-compatible)

```
WAITING | RUNNING | SWAPPED
FINISHED_STOPPED          → "stop"
FINISHED_LENGTH_CAPPED    → "length"
FINISHED_ABORTED          → "abort"
FINISHED_IGNORED          → "prompt_length_exceeded"
FINISHED_PREEMPTED        → "insufficient_system_resource"
```

## Scheduler policy (ported)

1. Reap finished running sequences, free blocks.
2. Schedule decode steps for running seqs under `max_num_seqs` / `max_num_batched_tokens`.
3. Admit waiting prefills when blocks + token budget allow.
4. `abort_request(request_id)` cancels waiting + running.

## DSpark ↔ Tesla

DeepInfer env knobs:
- `DEEPINFER_ENABLE_MTP_VARLEN=1`
- `DEEPINFER_USE_NEXTN_DECODE` (default draft len 5)

loopmoe CLI:
```bash
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" \
  --tesla --tesla-k 5 --tesla-tree-width 3
```

Confidence-scheduled verification lives in `tesla/verify.rs` (`schedule_verify_len_smart`).

## Not ported (by design)

- Full Python worker / VE multimodal pipeline
- Ascend NPU path
- TileLang / DeepSelect kernels (use Fertile + DoBu + Candle instead)
- Production KV service split (P/D disaggregation)

## References inside the DeepInfer tree

- `csrc/engine/scheduler.hpp`
- `csrc/engine/sequence.hpp`
- `csrc/engine/block_manager.hpp`
- `csrc/engine/mtp_varlen.hpp`
- `deepinfer/model_executor/`
- `vendor/DeepEP`, `vendor/DeepGEMM`, `vendor/FlashMLA`
