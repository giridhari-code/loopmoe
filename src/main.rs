use clap::Parser;
use loopmoe::{
    config::ModelConfig,
    engine::InferenceEngine,
    sampler::SamplingParams,
    tokenizer::{resolve_tokenizer_path, Tokenizer},
    tesla::TeslaConfig,
};
use std::path::PathBuf;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "sentinel", about = "Sentinel v1.0 — parallel Looped MoE + Tesla")]
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
    #[arg(long)]
    tesla: bool,
    #[arg(long, default_value_t = 5)]
    tesla_k: usize,
    /// Tree draft (parallel batch expand). Default on when --tesla.
    #[arg(long, default_value_t = true)]
    tesla_tree: bool,
    #[arg(long, default_value_t = 2)]
    tesla_tree_width: usize,
    #[arg(long, default_value_t = 0)]
    tesla_tree_depth: usize,
    /// Force sequential chain draft instead of parallel tree.
    #[arg(long)]
    tesla_chain: bool,
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
    let tok_path =
        resolve_tokenizer_path(std::path::Path::new(&args.model), args.tokenizer.as_deref());
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
        if !params.stop_token_ids.contains(&eos) {
            params.stop_token_ids.push(eos);
        }
    }
    let input_ids = tokenizer.encode(&args.prompt)?;
    tracing::info!(prompt_tokens = input_ids.len(), "encoded prompt");
    let out_ids = if args.tesla {
        let mut tcfg = TeslaConfig::with_nextn(args.tesla_k);
        if args.tesla_chain {
            tcfg.use_tree = false;
            tracing::info!("Tesla sequential chain draft");
        } else {
            tcfg.use_tree = args.tesla_tree;
            tcfg.tree_width = args.tesla_tree_width.max(1);
            tcfg.tree_depth = if args.tesla_tree_depth == 0 {
                args.tesla_k
            } else {
                args.tesla_tree_depth
            };
            tcfg.use_dobu_fuse = true;
            tracing::info!(
                width = tcfg.tree_width,
                depth = tcfg.tree_depth,
                "Tesla PARALLEL tree draft (batch expand)"
            );
        }
        engine.generate_tesla(&input_ids, args.max_tokens, &params, &tcfg)?
    } else {
        engine.generate(&input_ids, args.max_tokens, &params)?
    };
    let text = tokenizer.decode(&out_ids[input_ids.len()..])?;
    println!("{}", text);
    Ok(())
}
