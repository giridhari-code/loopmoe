# Sentinel v1.0

Looped MoE + Fertile + **Tesla** + DoBu + **parallel-first** design + multi-GPU mesh + training.

## Parallel (default) — not sequential

| Component | Parallel behaviour |
|-----------|-------------------|
| **Tesla tree draft** | Level-sync **batch forward** (all beam tokens in one matmul) + rayon branch expand |
| **Fertile MoE** | Active experts mapped **in parallel** (rayon); scatter reduce |
| **Chunk verify** | One sequence forward (not N token steps) |
| **DoBu GEMM** | Fused linear / QK / AV through DoBu |
| **Multi-GPU** | EP/TP/DP mesh + DeepEP-style all-to-all stubs |

Sequential chain draft only if you pass `--tesla-chain`.

```bash
# Parallel tree (default with --tesla)
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" --tesla --tesla-k 4

# Explicit tree width/depth
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" \
  --tesla --tesla-tree-width 3 --tesla-tree-depth 4

# Force old sequential chain
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" --tesla --tesla-chain
```

## Multi-GPU

```bash
export WORLD_SIZE=8 RANK=0 LOCAL_RANK=0 EP_SIZE=8 TP_SIZE=1
```

`ParallelConfig` / `DeviceMesh` / `ExpertParallelPlan` / `AllToAllStub` — wire DeepEP for real collectives.

## DoBu

DeepGEMM-style GEMM + FlashMLA-style MLA. Tesla `fuse_*` calls DoBu.

## Training

See TRAINING.md.
