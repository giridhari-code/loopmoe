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

## Checkpoint layout

Each save writes a directory:

```text
outputs/<stage>/step-N/
  model.safetensors      # weights
  optimizer.safetensors  # Adam m/v moments (keyed by param name)
  optim_meta.json        # step_t, betas, eps, weight_decay
  train_state.json       # training step + stage
```

### Resume

```json
{
  "stage": "pretrain",
  "resume_from": "outputs/pretrain/step-500",
  "max_steps": 1000
}
```

Restores weights + Adam moments + step counter. Training continues from `step` toward `max_steps`.

`init_from` = weights only (fresh optimizer).  
`resume_from` = full trajectory resume.

---

## Training optimisations (co-opt)

| Feature | Status | Notes |
|---------|--------|-------|
| **MoE aux loss** | ✅ | `avg_aux * moe_aux_coef` |
| **Grad accumulation** | ✅ | Sum micro-batch losses → one step |
| **DPO reference model** | ✅ | Frozen π_ref; proper log-prob ratio |
| **LR schedule** | ✅ | Linear warmup → cosine decay |
| **Grad clip** | ✅ | Global L2 `max_grad_norm` |
| **Optimizer state resume** | ✅ | `ResumableAdamW` + `resume_from` |

### LR schedule

```
warmup:  lr = peak * (step+1) / warmup_steps
cosine:  lr = min_lr + 0.5*(peak-min_lr)*(1+cos(π t))
```

### DPO formula

```
L = -log σ( β · ( [log πθ(y_w) − log πθ(y_l)] − [log πref(y_w) − log πref(y_l)] ) )
```

### Config knobs

```json
{
  "grad_accum": 8,
  "moe_aux_coef": 0.001,
  "batch_size": 1,
  "lr": 0.0003,
  "warmup_steps": 100,
  "max_grad_norm": 1.0,
  "min_lr_ratio": 0.1,
  "dpo_beta": 0.1,
  "resume_from": null
}
```
