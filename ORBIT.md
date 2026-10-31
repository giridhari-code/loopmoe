# Orbit architecture

Orbit is **loopmoe’s native** continuous-batch inference runtime.
It is not a wrapper around any external serving engine.

## Managers

| Manager | Role |
|---------|------|
| **OrbitSched** | Time — which sequences run, token budget, Tesla lane |
| **BlockLedger** | Space — physical pages, block tables, slot_mapping |
| **Pulse** (O2) | Execute — packed forward + sample |

## O1 — BlockLedger

```rust
use loopmoe::{BlockLedger, LedgerConfig, CacheKind};

let mut ledger = BlockLedger::new(LedgerConfig {
    num_pages: 2048,
    block_size: 16,
    kind: CacheKind::Mla, // or Gqa
    ..Default::default()
});

ledger.register_seq(1);
assert!(ledger.ensure_tokens(1, 128));
let slot = ledger.append_token_slot(1); // physical slot for next decode token
let map = ledger.build_slot_mapping(&[(1, 0, 128)]);
```

- Pages are **1-based** ids; free list is a stack.
- `SlotMapping.physical_slots` is what a paged-attention kernel indexes.
- Per-seq `RuntimeCaches` still hold tensors today; O2 binds a shared pool.

## O2 preview

`Pulse::step` will:

1. Take `SchedulerOutputs` + `SlotMapping`
2. Build `PackedBatch` (decode B>1)
3. One `model.forward_packed`
4. Sample + `ledger.append_token_slot`

## Integration

`OrbitSched` wraps `ContinuousScheduler` and keeps BlockLedger in sync on
admit / decode growth / finish / preempt.
