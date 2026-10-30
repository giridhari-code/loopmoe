use loopmoe::{config::ModelConfig, engine::InferenceEngine, sampler::SamplingParams, tokenizer::Tokenizer};

fn main() -> anyhow::Result<()> {
    let model = std::env::args().nth(1).expect("usage: offline_infer <weights_dir> [prompt]");
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "Hello".into());
    let engine = InferenceEngine::from_safetensors(&model, ModelConfig::sentinel_v1())?;
    let tok = Tokenizer::from_model_dir(&model).unwrap_or_else(|_| Tokenizer::byte_level());
    let ids = tok.encode(&prompt)?;
    let out = engine.generate(&ids, 32, &SamplingParams::greedy())?;
    println!("{}", tok.decode(&out[ids.len()..])?);
    Ok(())
}
