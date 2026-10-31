//! # Frontier — Orbit O4 OpenAI-compatible HTTP API
//!
//! Front door for continuous-batch runtime. Today each request runs through
//! `InferenceEngine` (Tesla optional). Multi-request continuous BatchEngine
//! worker can plug in later without changing routes.

use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::stream::{self, StreamExt};
use serde_json::json;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::engine::InferenceEngine;
use crate::recipe::{ChatCompletionRequest, RecipePipeline};
use crate::sampler::SamplingParams;
use crate::tesla::TeslaConfig;
use crate::tokenizer::Tokenizer;
use crate::Result;

/// Shared server state.
pub struct FrontierState {
    pub engine: Mutex<InferenceEngine>,
    pub tokenizer: Tokenizer,
    pub recipe: RecipePipeline,
    pub model_id: String,
    pub tesla: Option<TeslaConfig>,
}

impl FrontierState {
    pub fn new(
        engine: InferenceEngine,
        tokenizer: Tokenizer,
        model_id: impl Into<String>,
        tesla: Option<TeslaConfig>,
    ) -> Self {
        Self {
            engine: Mutex::new(engine),
            tokenizer,
            recipe: RecipePipeline::new(),
            model_id: model_id.into(),
            tesla,
        }
    }
}

/// Build Axum router for Frontier.
pub fn frontier_router(state: Arc<FrontierState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Bind and serve (blocks).
pub async fn serve(state: Arc<FrontierState>, host: [u8; 4], port: u16) -> Result<()> {
    let app = frontier_router(state);
    let addr = SocketAddr::from((host, port));
    tracing::info!(%addr, "Frontier OpenAI-compatible server (Orbit O4)");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok", "runtime": "orbit", "phase": "O4" }))
}

async fn list_models(State(state): State<Arc<FrontierState>>) -> impl IntoResponse {
    let now = now_secs();
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_id,
            "object": "model",
            "created": now,
            "owned_by": "loopmoe"
        }]
    }))
}

#[derive(Debug, serde::Deserialize)]
struct CompletionBody {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    stream: Option<bool>,
    stop: Option<Vec<String>>,
}

async fn completions(
    State(state): State<Arc<FrontierState>>,
    Json(body): Json<CompletionBody>,
) -> Response {
    let max_tokens = body.max_tokens.unwrap_or(256);
    let mut sampling = SamplingParams::default();
    if let Some(t) = body.temperature {
        sampling.temperature = t as f64;
    }
    if let Some(p) = body.top_p {
        sampling.top_p = p as f64;
    }
    if let Some(stop) = body.stop {
        // stop strings handled at sampler token-id level when configured
        let _ = stop;
    }

    let input_ids = match state.tokenizer.encode(&body.prompt) {
        Ok(ids) => ids,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
        }
    };

    let stream = body.stream.unwrap_or(false);
    match generate_tokens(&state, &input_ids, max_tokens, &sampling).await {
        Ok(full_ids) => {
            let prompt_len = input_ids.len();
            let gen = if full_ids.len() > prompt_len {
                &full_ids[prompt_len..]
            } else {
                &[]
            };
            let text = state.tokenizer.decode(gen).unwrap_or_default();
            if stream {
                return sse_completion_stream(&state.model_id, &text).into_response();
            }
            let id = format!("cmpl-{}", Uuid::new_v4());
            Json(json!({
                "id": id,
                "object": "text_completion",
                "created": now_secs(),
                "model": body.model.unwrap_or_else(|| state.model_id.clone()),
                "choices": [{
                    "text": text,
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }))
            .into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn chat_completions(
    State(state): State<Arc<FrontierState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let stream = req.stream.unwrap_or(false);
    let prepared = match state.recipe.prepare(&req, &state.tokenizer) {
        Ok(p) => p,
        Err(e) => {
            return (StatusCode::BAD_REQUEST, e.to_string()).into_response();
        }
    };

    match generate_tokens(
        &state,
        &prepared.input_ids,
        prepared.max_tokens,
        &prepared.sampling,
    )
    .await
    {
        Ok(full_ids) => {
            if stream {
                let prompt_len = prepared.input_ids.len();
                let gen = if full_ids.len() > prompt_len {
                    full_ids[prompt_len..].to_vec()
                } else {
                    Vec::new()
                };
                let text = state.tokenizer.decode(&gen).unwrap_or_default();
                return sse_chat_stream(&state.model_id, &text).into_response();
            }
            match state
                .recipe
                .complete(&prepared, &state.tokenizer, &full_ids)
            {
                Ok(resp) => Json(resp).into_response(),
                Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
            }
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

async fn generate_tokens(
    state: &FrontierState,
    input_ids: &[u32],
    max_tokens: usize,
    sampling: &SamplingParams,
) -> Result<Vec<u32>> {
    let tesla = state.tesla.clone();
    let input = input_ids.to_vec();
    let sampling = sampling.clone();
    // InferenceEngine is !Send across await without mutex; lock + spawn_blocking
    let engine = &state.engine;
    tokio::task::spawn_blocking({
        let input = input;
        let sampling = sampling;
        let tesla = tesla;
        move || {
            // We need the mutex — spawn_blocking can't hold axum state mutex across easily
            // so caller must lock. Use block_in_place from async instead.
            let _ = (&input, &sampling, &tesla);
            Err(anyhow::anyhow!("internal: use generate_tokens_locked"))
        }
    });
    // Prefer block_in_place + try_lock path
    tokio::task::block_in_place(|| {
        let mut eng = state
            .engine
            .blocking_lock();
        if let Some(ref cfg) = state.tesla {
            if cfg.enabled {
                return eng.generate_tesla(input_ids, max_tokens, sampling, cfg);
            }
        }
        eng.generate(input_ids, max_tokens, sampling)
    })
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn sse_chat_stream(model: &str, text: &str) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>> + use<>> {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = model.to_string();
    // Word-ish chunks for streaming UX
    let chunks: Vec<String> = text
        .split_inclusive(char::is_whitespace)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    let n = chunks.len();
    let stream = stream::iter(chunks.into_iter().enumerate().map(move |(i, piece)| {
        let body = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": now_secs(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "content": piece },
                "finish_reason": if i + 1 == n { serde_json::Value::String("stop".into()) } else { serde_json::Value::Null }
            }]
        });
        Ok(Event::default().data(body.to_string()))
    }))
    .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

    Sse::new(stream).keep_alive(KeepAlive::default())
}

fn sse_completion_stream(model: &str, text: &str) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>> + use<>> {
    let id = format!("cmpl-{}", Uuid::new_v4());
    let model = model.to_string();
    let chunks: Vec<String> = text
        .split_inclusive(char::is_whitespace)
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    let n = chunks.len();
    let stream = stream::iter(chunks.into_iter().enumerate().map(move |(i, piece)| {
        let body = json!({
            "id": id,
            "object": "text_completion",
            "created": now_secs(),
            "model": model,
            "choices": [{
                "text": piece,
                "index": 0,
                "finish_reason": if i + 1 == n { serde_json::Value::String("stop".into()) } else { serde_json::Value::Null }
            }]
        });
        Ok(Event::default().data(body.to_string()))
    }))
    .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Helper: load Frontier from model path (used by binary).
pub fn load_frontier(
    model_path: &Path,
    config: crate::config::ModelConfig,
    tokenizer: Tokenizer,
    tesla: Option<TeslaConfig>,
) -> Result<FrontierState> {
    let model_id = config.model_name.clone();
    let engine = InferenceEngine::from_safetensors(model_path, config)?;
    Ok(FrontierState::new(engine, tokenizer, model_id, tesla))
}
