//! Offline inference example
//!
//! Usage:
//!   cargo run --example offline_infer -- --model /path/to/model.safetensors

use clap::Parser;
use loopmoe::{config::ModelConfig, engine::InferenceEngine};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut config = ModelConfig::default();
    config.loop_count = 24;

    let engine = InferenceEngine::from_safetensors(&args.model, config)?;
    println!("Loaded on device: {:?}", engine.device);
    println!("Effective depth: {}", engine.config.effective_depth());
    Ok(())
}
