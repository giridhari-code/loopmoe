# Sentinel v1.0

Looped MoE + Fertile + **recipe** (deepseek-recipe-style API layer) + Rust training.

## Recipe (was missing vs deepseek-recipe)

| deepseek-recipe | Sentinel now |
|-----------------|--------------|
| Conversation / messages | `recipe::Conversation` |
| Prompt encoding | `SentinelEncoding` |
| Chat Completions | `ChatCompletionRequest/Response` |
| Pipeline | `RecipePipeline` |
| Example API server | `sentinel-server` (`--features server`) |

```bash
cargo run --release --example recipe_chat -- ./weights "Hello"

cargo run --release --features server --bin sentinel-server -- --model ./weights --port 8080
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Sentinel-v1.0","messages":[{"role":"user","content":"Hello"}]}'
```

Not ported (DeepSeek-specific): V4.1 encoding, images/OpenCV, tool-call blocks, Responses/Messages protocols — add when needed.
