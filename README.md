# Sentinel v1.0

Looped MoE + Fertile + **Tesla** (DSpark) + DoBu kernels + **single/multi-GPU parallel** + recipe API + Rust training.

## Tesla (was DSpark)

Confidence-scheduled speculative decoding:

```bash
# Chain draft + chunk verify
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" --tesla --tesla-k 5

# Tree draft (branched) + chunk verify + DoBu fuse
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" \
  --tesla --tesla-k 4 --tesla-tree --tesla-tree-width 2 --tesla-tree-depth 4
```

| Stage | What |
|--------|------|
| **Chain draft** | Single path, up to `k` tokens |
| **Tree draft** | Branch width × depth, survival-ranked best path |
| **Confidence** | Max-softmax + optional top-2 margin |
| **Schedule** | Survival product → verify length |
| **Chunk verify** | One sequence forward (heavy matmuls once) |
| **DoBu fuse** | `fuse_linear_*`, `fuse_qk_scores`, `fuse_av` via DoBu GEMM |
| **KV rewind** | Restore before rejected draft |

Stats: `drafted / scheduled / accepted / forwards / accept_rate`

## Parallelism (single + multi GPU)

| Mode | How |
|------|-----|
| **Single GPU** (default) | `ParallelConfig::single()` |
| **Expert Parallel** | `ep_size > 1` (DeepEP-style stubs) |
| **Tensor / Data / Hybrid** | `tp_size`, `dp_size`, mesh ranks |

```bash
export WORLD_SIZE=8 RANK=0 LOCAL_RANK=0 EP_SIZE=8 TP_SIZE=1
```

## DoBu kernels

- **GEMM** — DeepGEMM-style (Nn / Nt / Tn, grouped MoE, alpha/beta)
- **MLA** — FlashMLA-style prefill + KV-cache decode
- **Tesla fuse** — `tesla::fuse_*` calls DoBu for linear / attention matmuls

## Recipe / server

```bash
cargo run --release --features server --bin sentinel-server -- --model ./weights
```

## Training

See TRAINING.md.
