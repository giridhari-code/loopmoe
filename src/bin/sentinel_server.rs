use axum::{extract::State, routing::post, Json, Router};
use clap::Parser;
use loopmoe::{
    config::ModelConfig,
    engine::InferenceEngine,
    recipe::{ChatCompletionRequest, RecipePipeline},
    tokenizer::{resolve_tokenizer_path, Tokenizer},
};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
struct Args {
    #[arg(short, long)]
    model: String,
    #[arg(short, long)]
    config: Option<String>,
    #[arg(long)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value_t = 8080)]
    port: u16,
}

struct AppState {
    engine: InferenceEngine,
    tokenizer: Tokenizer,
    recipe: RecipePipeline,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();
    let args = Args::parse();
    let config = if let Some(p) = args.config { ModelConfig::from_json(&p)? } else { ModelConfig::sentinel_v1() };
    let engine = InferenceEngine::from_safetensors(&args.model, config)?;
    let tok_path = resolve_tokenizer_path(std::path::Path::new(&args.model), args.tokenizer.as_deref());
    let tokenizer = if tok_path.exists() { Tokenizer::from_path(&tok_path)? } else { Tokenizer::byte_level() };
    let state = Arc::new(AppState { engine, tokenizer, recipe: RecipePipeline::new() });
    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .layer(CorsLayer::permissive())
        .with_state(state);
    let addr = SocketAddr::from(([0, 0, 0, 0], args.port));
    tracing::info!(%addr, "Sentinel OpenAI-compatible server");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    let prepared = state.recipe.prepare(&req, &state.tokenizer)
        .map_err(|e| (axum::http::StatusCode::BAD_REQUEST, e.to_string()))?;
    let full_ids = tokio::task::block_in_place(|| {
        state.engine.generate(&prepared.input_ids, prepared.max_tokens, &prepared.sampling)
    }).map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let resp = state.recipe.complete(&prepared, &state.tokenizer, &full_ids)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    Ok(Json(serde_json::to_value(resp).unwrap()))
}
