use clap::{Parser, ValueEnum};
use loopmoe::config::ModelConfig;
use loopmoe::train::config::{TrainConfig, TrainStage};
use loopmoe::train::{dpo, midtrain, pretrain, sft};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(name = "sentinel-train", about = "Sentinel v1.0 training (pre / mid / sft / dpo)")]
struct Args {
    #[arg(short, long)]
    config: Option<String>,
    #[arg(long, value_enum)]
    stage: Option<StageArg>,
    #[arg(long)]
    data: Option<String>,
    #[arg(long)]
    init_from: Option<String>,
    #[arg(long)]
    output: Option<String>,
}

#[derive(Clone, Debug, ValueEnum)]
enum StageArg { Pretrain, Midtrain, Sft, Dpo }

fn select_device() -> candle_core::Device {
    if let Ok(d) = candle_core::Device::new_cuda(0) { return d; }
    tracing::warn!("CUDA unavailable — training on CPU");
    candle_core::Device::Cpu
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();
    let args = Args::parse();
    let mut cfg = if let Some(p) = &args.config {
        TrainConfig::from_json(p)?
    } else {
        TrainConfig::pretrain_defaults()
    };
    if let Some(s) = args.stage {
        cfg.stage = match s {
            StageArg::Pretrain => TrainStage::Pretrain,
            StageArg::Midtrain => TrainStage::Midtrain,
            StageArg::Sft => TrainStage::Sft,
            StageArg::Dpo => TrainStage::Dpo,
        };
    }
    if let Some(d) = args.data { cfg.data_path = d; }
    if let Some(i) = args.init_from { cfg.init_from = Some(i); }
    if let Some(o) = args.output { cfg.output_dir = o; }
    if cfg.model.model_name.is_empty() { cfg.model = ModelConfig::sentinel_v1(); }
    let device = select_device();
    tracing::info!(model = %cfg.model.model_name, stage = ?cfg.stage, device = ?device, "Sentinel train");
    match cfg.stage {
        TrainStage::Pretrain => pretrain::run(&cfg, &device)?,
        TrainStage::Midtrain => midtrain::run(&cfg, &device)?,
        TrainStage::Sft => sft::run(&cfg, &device)?,
        TrainStage::Dpo => dpo::run(&cfg, &device)?,
    }
    Ok(())
}
