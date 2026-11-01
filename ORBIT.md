# Orbit architecture

Native continuous-batch runtime for LoopedMoE + Tesla (no external serving engine).

## Phases

| Phase | Status |
|-------|--------|
| O1 BlockLedger | Done |
| O2 / O2.1 PackedBatch B>1 | Done |
| O3 Tesla + chunked prefill | Done |
| O4 Frontier HTTP | Done |
| **O5 Prefix hash + EP packing** | **Done** |

## O5 Prefix cache

```rust
ledger.allocate_with_prefix(seq_id, &prompt_tokens); // hits shared pages
// ... run prefill for unique pages only (engine still fills KV) ...
ledger.seal_seq_prompt(seq_id, &prompt_tokens);
```

- FNV-1a chained hashes per page (`hash_token_block` / `hash_prompt_pages`)
- `prefix_index: HashMap<hash, page_id>` + refcount
- Stats: `ledger.stats.prefix.{hits,misses,pages_shared,pages_sealed}`

## O5 Expert Parallel

```rust
let (packed, meta) = pack_by_ep_rank(&tokens, &topk_ids, n_routed, ep_size)?;
// local experts on each rank's slice...
let out = unpack_combine(&expert_out, &CombineMeta::from_dispatch(&meta))?;
```

- `expert_to_rank` matches `ExpertParallelPlan` contiguous shards
- `AllToAllStub::dispatch` packs when `ep_size > 1` and top-k provided
- **NCCL/DeepEP device transfer** still not linked (same-process logical EP)

## Layout

```
src/orbit/prefix.rs   # hash helpers
src/orbit/ledger.rs   # allocate_with_prefix / seal
src/parallel/collectives.rs  # pack_by_ep_rank / unpack
```
