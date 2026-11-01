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

Checkpoints: `outputs/<stage>/final/model.safetensors`

---

## Training optimisations (co-opt)

| Feature | Status | Notes |
|---------|--------|-------|
| **MoE aux loss** | ✅ | Load-balance term `avg_aux * moe_aux_coef` added to CE |
| **Grad accumulation** | ✅ | `grad_accum` in config (default 1). Loss scaled by `1/accum` |
| DPO reference model | ❌ | Still policy-only approximate |
| Grad clip / LR schedule | ❌ | Planned |
| Optimizer state resume | ❌ | Weights only |

### Config knobs

```json
{
  "grad_accum": 8,
  "moe_aux_coef": 0.001,
  "batch_size": 1,
  "lr": 0.0003
}
```

Effective batch size ≈ `batch_size * grad_accum`.
