use super::types::{Conversation, Role};

#[derive(Debug, Clone)]
pub struct RenderedPrompt {
    pub prompt: String,
    pub generation_start: usize,
}

pub struct SentinelEncoding;

impl SentinelEncoding {
    pub fn new() -> Self { Self }
    pub fn render_conversation(&self, conv: &Conversation) -> RenderedPrompt {
        render_conversation(conv)
    }
}

impl Default for SentinelEncoding {
    fn default() -> Self { Self::new() }
}

pub fn render_conversation(conv: &Conversation) -> RenderedPrompt {
    let mut prompt = String::new();
    for msg in &conv.messages {
        match msg.role {
            Role::System => { prompt.push_str("<|system|>\n"); prompt.push_str(msg.content.trim()); prompt.push('\n'); }
            Role::User => { prompt.push_str("<|user|>\n"); prompt.push_str(msg.content.trim()); prompt.push('\n'); }
            Role::Assistant => { prompt.push_str("<|assistant|>\n"); prompt.push_str(msg.content.trim()); prompt.push('\n'); }
            Role::Tool => { prompt.push_str("<|tool|>\n"); prompt.push_str(msg.content.trim()); prompt.push('\n'); }
        }
    }
    prompt.push_str("<|assistant|>\n");
    let generation_start = prompt.len();
    RenderedPrompt { prompt, generation_start }
}
