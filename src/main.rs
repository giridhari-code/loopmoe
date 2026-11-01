use clap::Parser;
use loopmoe::{
    config::ModelConfig,
    engine::InferenceEngine,
    sampler::SamplingParams,
    scheduler::SchedulerConfig,
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
    /// Path to a text file with one prompt per line (enables multi-request batch).
    #[arg(long)]
    prompts: Option<PathBuf>,
    /// Force continuous-batch multi-request path even for a single --prompt.
    #[arg(long)]
    batch: bool,
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
    #[arg(long, default_value_t = true)]
    tesla_tree: bool,
    #[arg(long, default_value_t = 2)]
    tesla_tree_width: usize,
    #[arg(long, default_value_t = 0)]
    tesla_tree_depth: usize,
    #[arg(long)]
    tesla_chain: bool,
    /// GPU KV blocks for batch scheduler (DeepInfer-style).
    #[arg(long, default_value_t = 2048)]
    num_gpu_blocks: usize,
    #[arg(long, default_value_t = 16)]
    block_size: usize,
    #[arg(long, default_value_t = 256)]
    max_num_seqs: usize,
    #[arg(long, default_value_t = 8192)]
    max_num_batched_tokens: usize,
}

fn build_tesla(args: &Args) -> TeslaConfig {
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
        tcfg.enable_mtp_varlen = true;
        tracing::info!(
            width = tcfg.tree_width,
            depth = tcfg.tree_depth,
            nextn = tcfg.effective_nextn(),
            "Tesla PARALLEL tree draft + MTP varlen"
        );
    }
    tcfg
}

fn load_prompts(args: &Args) -> anyhow::Result<Vec<String>> {
    if let Some(path) = &args.prompts {
        let text = std::fs::read_to_string(path)?;
        let lines: Vec<String> = text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        if lines.is_empty() {
            anyhow::bail!("no prompts in {}", path.display());
        }
        Ok(lines)
    } else {
        Ok(vec![args.prompt.clone()])
    }
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

    let prompts = load_prompts(&args)?;
    let use_batch = args.batch || prompts.len() > 1 || args.prompts.is_some();

    if use_batch {
        // ---- Multi-request continuous batch path ----
        tracing::info!(n = prompts.len(), "multi-request batch path");
        let mut requests: Vec<(Vec<u32>, usize)> = Vec::with_capacity(prompts.len());
        for p in &prompts {
            let ids = tokenizer.encode(p)?;
            requests.push((ids, args.max_tokens));
        }
        let tesla = if args.tesla {
            Some(build_tesla(&args))
        } else {
            None
        };
        let sched = SchedulerConfig {
            max_num_seqs: args.max_num_seqs,
            max_num_batched_tokens: args.max_num_batched_tokens,
        };
        let results = engine.generate_batch(
            &requests,
            &params,
            tesla,
            sched,
            args.num_gpu_blocks,
            args.block_size,
        )?;

        for i in 0..prompts.len() {
            if let Some(out_ids) = results.get(&i) {
                let prompt_len = requests[i].0.len();
                let gen = if out_ids.len() > prompt_len {
                    tokenizer.decode(&out_ids[prompt_len..])?
                } else {
                    String::new()
                };
                println!("=== [{}] ===", i);
                println!("{}", gen);
            } else {
                tracing::warn!(i, "missing result for request");
            }
        }
    } else {
        // ---- Single-request path ----
        let input_ids = tokenizer.encode(&prompts[0])?;
        tracing::info!(prompt_tokens = input_ids.len(), "encoded prompt");
        let out_ids = if args.tesla {
            let tcfg = build_tesla(&args);
            engine.generate_tesla(&input_ids, args.max_tokens, &params, &tcfg)?
        } else {
            engine.generate(&input_ids, args.max_tokens, &params)?
        };
        let text = tokenizer.decode(&out_ids[input_ids.len()..])?;
        println!("{}", text);
    }
    Ok(())
}
