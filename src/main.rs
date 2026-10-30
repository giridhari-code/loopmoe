use clap::Parser;
use loopmoe::{
    config::ModelConfig,
    engine::InferenceEngine,
    sampler::SamplingParams,
    tokenizer::{resolve_tokenizer_path, Tokenizer},
};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "sentinel", about = "Sentinel v1.0 — Looped MoE inference (Fertile)")]
struct Args {
    #[arg(short, long)]
    model: String,
    #[arg(short, long)]
    config: Option<String>,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(short, long, default_value = "Hello")]
    prompt: String,
    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    top_p: Option<f64>,
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f64,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, value_delimiter = ',')]
    stop: Vec<u32>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();
    let args = Args::parse();
    let config = if let Some(cfg_path) = args.config {
        ModelConfig::from_json(&cfg_path)?
    } else {
        ModelConfig::sentinel_v1()
    };
    tracing::info!(model = %config.model_name, loop_count = config.loop_count, "Loading Sentinel");
    let engine = InferenceEngine::from_safetensors(&args.model, config)?;
    let tok_path = resolve_tokenizer_path(std::path::Path::new(&args.model), args.tokenizer.as_deref());
    let tokenizer = if tok_path.exists() {
        Tokenizer::from_path(&tok_path)?
    } else {
        tracing::warn!(?tok_path, "tokenizer missing — byte-level fallback");
        Tokenizer::byte_level()
    };
    let mut params = SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        stop_token_ids: args.stop,
        seed: args.seed,
    };
    if let Some(eos) = tokenizer.eos_id() {
        if !params.stop_token_ids.contains(&eos) { params.stop_token_ids.push(eos); }
    }
    let input_ids = tokenizer.encode(&args.prompt)?;
    tracing::info!(prompt_tokens = input_ids.len(), "encoded prompt");
    let out_ids = engine.generate(&input_ids, args.max_tokens, &params)?;
    let text = tokenizer.decode(&out_ids[input_ids.len()..])?;
    println!("{}", text);
    Ok(())
}
