# Sentinel v1.0

Looped Transformer + Shared/Routed MoE inference.

| | |
|--|--|
| **Model** | `Sentinel-v1.0` |
| **Kernel** | Fertile (CUDA → ARM → CPU) |
| **Loop count** | 24 |
| **MoE** | DeepInfer-style routed + shared experts |

Not a DeepSeek-V4.1 clone — MoE **config/weight naming** follows DeepInfer HF conventions; architecture is Sentinel (looped stack).

```bash
cargo run --release --bin sentinel -- --model /path/to/weights.safetensors
cargo run --release --bin sentinel -- --model ./weights --config configs/sentinel-v1.0.json
```

Shared experts: `mlp.shared_experts.{i}` or `mlp.experts.{n_routed+i}`.
