# Sentinel v1.0

Looped MoE + Fertile + **Tesla** (DSpark) + recipe API + Rust training.

## Tesla (was DSpark)

Confidence-scheduled speculative decoding:

```bash
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" --tesla --tesla-k 5
```

- Draft up to `k` tokens
- Confidence = softmax max (proxy head)
- Schedule verify length by survival product
- Verify + KV rewind (`restore_len`)

## Recipe / server

```bash
cargo run --release --features server --bin sentinel-server -- --model ./weights
```

## Training

See TRAINING.md.
