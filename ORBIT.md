# Orbit architecture

Orbit is **loopmoe’s native** continuous-batch inference runtime.

## Managers

| Manager | Role |
|---------|------|
| **OrbitSched** | Time — decode / Tesla / chunked prefill lanes |
| **BlockLedger** | Space — physical pages, slot_mapping |
| **Pulse** | Execute — PackedBatch B>1 decode |

## Phases

| Phase | Status |
|-------|--------|
| O1 BlockLedger | Done |
| O2 PackedBatch + Pulse | Done |
| O2.1 B>1 decode | Done |
| **O3 Tesla-in-schedule + chunked prefill** | **Done** |
| O4 Frontier server | Planned |
| O5 Prefix hash + EP | Planned |

## O3 details

### Chunked prefill

```text
max_num_batched_tokens budget
  └── each seq contributes min(remaining_prompt, prefill_chunk_size)
```

- `SchedulerConfig.prefill_chunk_size` (default **512**)
- Continuing prefills preferred over new admits
- Only the **last** chunk samples the first generated token; earlier chunks only fill KV

### Tesla lane

```text
SchedulerOutputs {
  prefill_chunks,  // chunked prompt work
  decode_ids,      // plain 1-token decode
  tesla_ids,       // speculative (budget = draft_len)
}
```

- Seq with `draft_len > 0` → Tesla lane
- Token budget uses `draft_len` (not 1)
- BatchEngine runs Tesla only for `tesla_ids`

## Config knobs

```rust
SchedulerConfig {
    max_num_seqs: 256,
    max_num_batched_tokens: 8192,
    prefill_chunk_size: 512,
    tesla_draft_estimate: 5,
    shortest_first: true,
    enable_preempt: true,
}
```
