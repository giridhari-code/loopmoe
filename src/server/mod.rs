//! OpenAI-compatible HTTP surface (delegates to Frontier / Orbit O4).

pub mod routes;
pub mod openai_types;
pub mod sse;
pub mod middleware;

pub use routes::create_router;
