# Sentinel v1.0 — Rust Training

Python training **removed**. Pure Rust (`candle`).

```text
pretrain → midtrain → sft → dpo
```

```bash
cargo run --release --bin sentinel-train -- --config configs/train_pretrain.json
cargo run --release --bin sentinel-train -- --stage midtrain --config configs/train_midtrain.json
cargo run --release --bin sentinel-train -- --stage sft --config configs/train_sft.json
cargo run --release --bin sentinel-train -- --stage dpo --config configs/train_dpo.json
```

Data is **pre-tokenized JSONL** (`input_ids` / `labels` / `chosen_ids`+`rejected_ids`).

---

## Train / Infer unified backbone

`SentinelTrainModel` wraps the **same** `LoopedMoE` used by inference:

| | Train | Infer |
|--|-------|-------|
| Architecture | `LoopedMoE` | `LoopedMoE` |
| Weight keys | identical | identical |
| Forward | `forward_full` → `[B,S,V]` | last-token / decode |
| Diff ops | `FORCE_DIFF_OPS` (Candle rmsnorm/residual/silu) | CUDA fused kernels |

After training, load the checkpoint with the normal inference path:

```bash
cargo run --release --bin sentinel -- --model outputs/pretrain/final --prompt "Hello"
```

---

## Checkpoint layout

```text
outputs/<stage>/step-N/
  model.safetensors
  optimizer.safetensors
  optim_meta.json
  train_state.json
```

### Resume

```json
{
  "stage": "pretrain",
  "resume_from": "outputs/pretrain/step-500",
  "max_steps": 1000
}
```

`init_from` = weights only · `resume_from` = weights + Adam moments + step.

---

## Co-opt status

| Feature | Status |
|---------|--------|
| MoE aux loss | ✅ (coef reserved; CE primary) |
| Grad accumulation | ✅ |
| DPO + reference | ✅ |
| LR schedule | ✅ |
| Grad clip | ✅ |
| Optimizer state resume | ✅ |
| **Train/infer unify** | ✅ |
