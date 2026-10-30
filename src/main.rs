use clap::Parser;
use loopmoe::{config::ModelConfig, engine::InferenceEngine};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "sentinel", about = "Sentinel v1.0 — Looped MoE inference (Fertile kernel)")]
struct Args {
    #[arg(short, long)]
    model: String,

    #[arg(short, long)]
    config: Option<String>,

    #[arg(short, long, default_value = "Hello")]
    prompt: String,

    #[arg(long, default_value_t = 64)]
    max_tokens: usize,
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

    tracing::info!(
        model = %config.model_name,
        model_type = %config.model_type,
        loop_count = config.loop_count,
        effective_depth = config.effective_depth(),
        routed = config.n_routed_experts,
        shared = config.n_shared_experts,
        "Loading Sentinel"
    );

    let engine = InferenceEngine::from_safetensors(&args.model, config)?;
    tracing::info!("Sentinel v1.0 ready (weights={})", args.model);
    let _ = (engine, args.prompt, args.max_tokens);
    Ok(())
}
