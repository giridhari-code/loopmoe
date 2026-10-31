//! # Frontier — Orbit O4 OpenAI-compatible HTTP API
//!
//! Routes request traffic onto `InferenceEngine` (optional Tesla).
//! Continuous BatchEngine can replace the generate path later without route changes.

use std::convert::Infallible;
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
use futures::stream;
use serde::Deserialize;
use serde_json::json;
use tokio::sync::Mutex;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

use crate::config::ModelConfig;
use crate::engine::InferenceEngine;
use crate::recipe::{ChatCompletionRequest, RecipePipeline};
use crate::sampler::SamplingParams;
use crate::tesla::TeslaConfig;
use crate::tokenizer::Tokenizer;
use crate::Result;

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

pub fn frontier_router(state: Arc<FrontierState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

pub async fn serve(state: Arc<FrontierState>, host: [u8; 4], port: u16) -> Result<()> {
    let app = frontier_router(state);
    let addr = SocketAddr::from((host, port));
    tracing::info!(%addr, "Frontier server (Orbit O4)");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

pub fn load_frontier(
    model_path: &Path,
    config: ModelConfig,
    tokenizer: Tokenizer,
    tesla: Option<TeslaConfig>,
) -> Result<FrontierState> {
    let model_id = config.model_name.clone();
    let engine = InferenceEngine::from_safetensors(model_path, config)?;
    Ok(FrontierState::new(engine, tokenizer, model_id, tesla))
}

async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok", "runtime": "orbit", "phase": "O4-frontier" }))
}

async fn list_models(State(state): State<Arc<FrontierState>>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_id,
            "object": "model",
            "created": now_secs(),
            "owned_by": "loopmoe"
        }]
    }))
}

#[derive(Debug, Deserialize)]
struct CompletionBody {
    #[serde(default)]
    model: Option<String>,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    stream: Option<bool>,
}

async fn completions(
    State(state): State<Arc<FrontierState>>,
    Json(body): Json<CompletionBody>,
) -> Response {
    let max_tokens = body.max_tokens.unwrap_or(256);
    let mut sampling = SamplingParams::default();
    if let Some(t) = body.temperature {
        sampling.temperature = t;
    }
    if let Some(p) = body.top_p {
        sampling.top_p = p;
    }

    let input_ids = match state.tokenizer.encode(&body.prompt) {
        Ok(ids) => ids,
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };

    match generate(&state, &input_ids, max_tokens, &sampling).await {
        Ok(full_ids) => {
            let prompt_len = input_ids.len();
            let gen = if full_ids.len() > prompt_len {
                &full_ids[prompt_len..]
            } else {
                &[]
            };
            let text = state.tokenizer.decode(gen).unwrap_or_default();
            if body.stream.unwrap_or(false) {
                return completion_sse(&state.model_id, &text).into_response();
            }
            Json(json!({
                "id": format!("cmpl-{}", Uuid::new_v4()),
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
        Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
    };

    match generate(
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
                return chat_sse(&state.model_id, &text).into_response();
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

async fn generate(
    state: &FrontierState,
    input_ids: &[u32],
    max_tokens: usize,
    sampling: &SamplingParams,
) -> Result<Vec<u32>> {
    let input_ids = input_ids.to_vec();
    let sampling = sampling.clone();
    let tesla = state.tesla.clone();

    tokio::task::block_in_place(|| {
        let mut eng = state.engine.blocking_lock();
        if let Some(ref cfg) = tesla {
            if cfg.enabled {
                return eng.generate_tesla(&input_ids, max_tokens, &sampling, cfg);
            }
        }
        eng.generate(&input_ids, max_tokens, &sampling)
    })
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn chat_sse(
    model: &str,
    text: &str,
) -> Sse<stream::Iter<std::vec::IntoIter<std::result::Result<Event, Infallible>>>> {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let model = model.to_string();
    let mut events = Vec::new();
    let pieces: Vec<&str> = text
        .split_inclusive(char::is_whitespace)
        .filter(|s| !s.is_empty())
        .collect();
    let n = pieces.len();
    for (i, piece) in pieces.into_iter().enumerate() {
        let body = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": now_secs(),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": { "content": piece },
                "finish_reason": if i + 1 == n { json!("stop") } else { json!(null) }
            }]
        });
        events.push(Ok(Event::default().data(body.to_string())));
    }
    if n == 0 {
        let body = json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": now_secs(),
            "model": model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": "stop" }]
        });
        events.push(Ok(Event::default().data(body.to_string())));
    }
    events.push(Ok(Event::default().data("[DONE]")));
    Sse::new(stream::iter(events)).keep_alive(KeepAlive::default())
}

fn completion_sse(
    model: &str,
    text: &str,
) -> Sse<stream::Iter<std::vec::IntoIter<std::result::Result<Event, Infallible>>>> {
    let id = format!("cmpl-{}", Uuid::new_v4());
    let model = model.to_string();
    let mut events = Vec::new();
    let pieces: Vec<&str> = text
        .split_inclusive(char::is_whitespace)
        .filter(|s| !s.is_empty())
        .collect();
    let n = pieces.len();
    for (i, piece) in pieces.into_iter().enumerate() {
        let body = json!({
            "id": id,
            "object": "text_completion",
            "created": now_secs(),
            "model": model,
            "choices": [{
                "text": piece,
                "index": 0,
                "finish_reason": if i + 1 == n { json!("stop") } else { json!(null) }
            }]
        });
        events.push(Ok(Event::default().data(body.to_string())));
    }
    events.push(Ok(Event::default().data("[DONE]")));
    Sse::new(stream::iter(events)).keep_alive(KeepAlive::default())
}
