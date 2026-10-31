//! OpenAI-compatible HTTP server
//!
//! Routes: /v1/completions, /v1/chat/completions
//! Streaming via SSE, auth + rate-limit middleware

pub mod routes;
pub mod openai_types;
pub mod sse;
pub mod middleware;

pub use routes::create_router;
