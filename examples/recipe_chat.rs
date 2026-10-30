use loopmoe::{
    config::ModelConfig,
    engine::InferenceEngine,
    recipe::{ChatCompletionRequest, RecipePipeline},
    tokenizer::Tokenizer,
};

fn main() -> anyhow::Result<()> {
    let model = std::env::args().nth(1).expect("usage: recipe_chat <weights> [msg]");
    let msg = std::env::args().nth(2).unwrap_or_else(|| "Hello".into());
    let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "Sentinel-v1.0",
        "messages": [{"role": "user", "content": msg}],
        "max_tokens": 32,
        "temperature": 0.0
    }))?;
    let engine = InferenceEngine::from_safetensors(&model, ModelConfig::sentinel_v1())?;
    let tok = Tokenizer::from_model_dir(&model).unwrap_or_else(|_| Tokenizer::byte_level());
    let recipe = RecipePipeline::new();
    let prepared = recipe.prepare(&req, &tok)?;
    println!("--- prompt ---\n{}", prepared.prompt);
    let out = engine.generate(&prepared.input_ids, prepared.max_tokens, &prepared.sampling)?;
    let resp = recipe.complete(&prepared, &tok, &out)?;
    println!("--- response ---\n{}", serde_json::to_string_pretty(&resp)?);
    Ok(())
}
