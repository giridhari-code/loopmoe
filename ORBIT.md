# Orbit architecture

Orbit is **loopmoe’s native** continuous-batch inference runtime.

## Managers

| Manager | Role | Phase |
|---------|------|-------|
| **OrbitSched** | Time — who runs this step | O1 |
| **BlockLedger** | Space — physical pages, slot_mapping | O1 |
| **Pulse** | Execute — PackedBatch → forward → sample | **O2** |

## O2 — PackedBatch + Pulse

```rust
use loopmoe::{PackedBatch, Pulse, PackRole, PackItem};

let pack = PackedBatch { items: vec![/* decode + prefill */], .. };
let pulse = Pulse::new(&model, &device);
let out = pulse.run(&pack, &mut caches, &mut generated, &mut rngs, &sampling)?;
// out.new_tokens: seq_id → tokens this step
```

**Today:** one `model.forward` per `PackItem` (each seq has batch=1 KV/MLA cache).
**Next (O2.1):** shared page pool + `forward_packed` for true B>1 decode.

`BatchEngine::step` already:
1. `schedule()`
2. builds `PackedBatch` + ledger `SlotMapping`
3. runs `Pulse` for plain prefill/decode
4. Tesla path remains specialized per-seq

## Layout

```
src/orbit/
├── ledger.rs   # O1 BlockLedger
├── sched.rs    # OrbitSched
├── pack.rs     # O2 PackedBatch
├── pulse.rs    # O2 Pulse
└── core.rs     # OrbitCore helper
```
