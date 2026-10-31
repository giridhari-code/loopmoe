//! Frontier HTTP server binary (Orbit O4).

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use loopmoe::{
    config::ModelConfig,
    frontier::{load_frontier, serve},
    tesla::TeslaConfig,
    tokenizer::{resolve_tokenizer_path, Tokenizer},
};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "sentinel-server", about = "Orbit Frontier OpenAI-compatible server")]
struct Args {
    #[arg(short, long)]
    model: String,

    #[arg(short, long)]
    config: Option<String>,

    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(long, default_value_t = 8080)]
    port: u16,

    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Enable Tesla speculative decoding for all requests.
    #[arg(long, default_value_t = false)]
    tesla: bool,

    #[arg(long, default_value_t = 5)]
    tesla_k: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let args = Args::parse();
    let config = if let Some(p) = &args.config {
        ModelConfig::from_json(p)?
    } else {
        ModelConfig::sentinel_v1()
    };

    let tok_path = resolve_tokenizer_path(std::path::Path::new(&args.model), args.tokenizer.as_deref());
    let tokenizer = if tok_path.exists() {
        Tokenizer::from_path(&tok_path)?
    } else {
        tracing::warn!("tokenizer.json missing — byte-level fallback");
        Tokenizer::byte_level()
    };

    let tesla = if args.tesla {
        let mut t = TeslaConfig::default();
        t.enabled = true;
        t.max_draft = args.tesla_k.max(1);
        Some(t)
    } else {
        None
    };

    let state = Arc::new(load_frontier(
        std::path::Path::new(&args.model),
        config,
        tokenizer,
        tesla,
    )?);

    let host = parse_host(&args.host)?;
    serve(state, host, args.port).await?;
    Ok(())
}

fn parse_host(s: &str) -> anyhow::Result<[u8; 4]> {
    if s == "0.0.0.0" || s == "*" {
        return Ok([0, 0, 0, 0]);
    }
    if s == "localhost" || s == "127.0.0.1" {
        return Ok([127, 0, 0, 1]);
    }
    let parts: Vec<_> = s.split('.').collect();
    if parts.len() != 4 {
        anyhow::bail!("invalid host {s}");
    }
    Ok([
        parts[0].parse()?,
        parts[1].parse()?,
        parts[2].parse()?,
        parts[3].parse()?,
    ])
}
