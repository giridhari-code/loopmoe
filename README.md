# Sentinel v1.0

Looped MoE + Fertile inference. **Training is pure Rust** (`sentinel-train`).

```bash
# inference
cargo run --release --bin sentinel -- --model ./weights

# training
cargo run --release --bin sentinel-train -- --config configs/train_pretrain.json
```

See [TRAINING.md](TRAINING.md). Python pipeline removed.
