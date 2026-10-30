//! Sentinel recipe layer (inspired by deepseek-ai/deepseek-recipe).
//! ChatCompletionRequest → Conversation → prompt → token ids → response.

pub mod types;
pub mod chat_template;
pub mod openai;
pub mod pipeline;

pub use types::{Conversation, Message, Role};
pub use chat_template::{render_conversation, SentinelEncoding};
pub use openai::{ChatCompletionChoice, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatRole};
pub use pipeline::{PreparedRequest, RecipePipeline};
