# Sentinel v1.0

Looped MoE + Fertile inference. Training: pure Rust (`sentinel-train`).

## Inference

```bash
cargo run --release --bin sentinel -- \
  --model /path/to/weights \
  --prompt "Hello" \
  --max-tokens 64 \
  --temperature 0.7 \
  --top-p 0.9 \
  --top-k 50
```

Tokenizer, sampling (greedy/temp/top-k/top-p), stop tokens, seed — included.

## Training

See [TRAINING.md](TRAINING.md).
