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
