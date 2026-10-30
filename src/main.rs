use clap::Parser;
use loopmoe::{config::ModelConfig, engine::InferenceEngine};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "loopmoe", about = "MoE + Looped Transformer Inference Engine")]
struct Args {
    /// Path to safetensors weights (file or directory)
    #[arg(short, long)]
    model: String,

    /// Path to config.json (optional, uses defaults if missing)
    #[arg(short, long)]
    config: Option<String>,

    /// Prompt text
    #[arg(short, long, default_value = "Hello")]
    prompt: String,

    /// Max new tokens
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
        let mut cfg = ModelConfig::default();
        cfg.loop_count = 24; // as requested
        cfg
    };

    tracing::info!(
        "Effective depth = {} layers × {} loops = {}",
        config.num_layers,
        config.loop_count,
        config.effective_depth()
    );

    let engine = InferenceEngine::from_safetensors(&args.model, config)?;

    // Placeholder generation (tokenizer not wired yet)
    tracing::info!("Model loaded successfully on {:?}", engine.device);
    tracing::info!("Ready for inference. Tokenizer + full generation coming next.");

    Ok(())
}
