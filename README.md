# Sentinel v1.0

Looped MoE + Fertile + **Tesla** (DSpark) + DoBu kernels + recipe API + Rust training.

## Tesla (was DSpark)

Confidence-scheduled speculative decoding:

```bash
cargo run --release --bin sentinel -- --model ./weights --prompt "Hi" --tesla --tesla-k 5
```

- Draft up to `k` tokens
- Confidence = max-softmax (GPU-friendly, no full vocab `to_vec`)
- Optional top-2 margin for smarter scheduling
- Survival-product verify length
- Verify + KV rewind (`restore_len`)
- Stats: drafted / scheduled / accepted + acceptance rate

## DoBu kernels

- **GEMM** (`dobu::gemm`) — DeepGEMM-style interface (Nn / Nt / Tn, grouped MoE GEMM)
- **MLA** (`dobu::mla`) — FlashMLA-style prefill + KV-cache decode

Current backend is pure Candle (portable). Signatures are ready for a future CUDA/FFI drop-in of DeepGEMM / FlashMLA.

## Recipe / server

```bash
cargo run --release --features server --bin sentinel-server -- --model ./weights
```

## Training

See TRAINING.md.
