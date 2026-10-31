# Orbit architecture

Orbit is **loopmoe’s native** continuous-batch inference runtime.

## Managers

| Manager | Role | Phase |
|---------|------|-------|
| **OrbitSched** | Time — who runs this step | O1 |
| **BlockLedger** | Space — physical pages, slot_mapping | O1 |
| **Pulse** | Execute — PackedBatch → forward → sample | O2 / **O2.1** |

## O2.1 — True B>1 decode

```rust
// LoopedMoE
let logits = model.forward_decode_batch(&tokens, &positions, &mut cache_refs)?;
// logits: [B, vocab]
```

| Op | Batched? |
|----|----------|
| Embed | **Yes** `[B,1]` |
| RMSNorm / residual | **Yes** |
| MoE (Fertile) | **Yes** (flat `B*S`) |
| Final norm + lm_head | **Yes** `[B,H]→[B,V]` |
| Attention + RoPE + KV/MLA cache | Per-seq (different pos/len) |

Pulse groups all `PackRole::Decode` items into one `forward_decode_batch` call.
Prefill stays one-forward-per-seq (variable length).

## Layout

```
src/orbit/
├── ledger.rs   # O1
├── sched.rs
├── pack.rs     # O2
├── pulse.rs    # O2 + O2.1 batch decode
└── core.rs
```
