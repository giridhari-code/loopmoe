//! Router factory — prefers Frontier when state is available.

use axum::{
    routing::get,
    Router,
};

/// Placeholder router without engine state (tests / health only).
/// Production binary uses `loopmoe::frontier::frontier_router`.
pub fn create_router() -> Router {
    Router::new().route("/health", get(|| async { "ok" }))
}
