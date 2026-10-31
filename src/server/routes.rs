//! /v1/completions and /v1/chat/completions handlers

use axum::{
    routing::{get, post},
    Router,
};

/// Build the main API router
pub fn create_router() -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/completions", post(completions))
        .route("/v1/chat/completions", post(chat_completions))
}

async fn completions() -> &'static str {
    // TODO: wire to scheduler + engine
    "not implemented"
}

async fn chat_completions() -> &'static str {
    // TODO: wire to scheduler + engine
    "not implemented"
}
