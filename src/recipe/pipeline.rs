use anyhow::Result;
use crate::sampler::SamplingParams;
use crate::tokenizer::Tokenizer;
use super::chat_template::SentinelEncoding;
use super::openai::{ChatCompletionRequest, ChatCompletionResponse};
use super::types::Conversation;

#[derive(Debug, Clone)]
pub struct PreparedRequest {
    pub model: String,
    pub conversation: Conversation,
    pub prompt: String,
    pub input_ids: Vec<u32>,
    pub sampling: SamplingParams,
    pub max_tokens: usize,
}

pub struct RecipePipeline {
    encoding: SentinelEncoding,
}

impl Default for RecipePipeline {
    fn default() -> Self { Self::new() }
}

impl RecipePipeline {
    pub fn new() -> Self {
        Self { encoding: SentinelEncoding::new() }
    }

    pub fn prepare(&self, req: &ChatCompletionRequest, tokenizer: &Tokenizer) -> Result<PreparedRequest> {
        let conversation = req.to_conversation();
        let rendered = self.encoding.render_conversation(&conversation);
        let input_ids = tokenizer.encode(&rendered.prompt)?;
        let mut sampling = SamplingParams::default();
        sampling.temperature = req.temperature.unwrap_or(0.0);
        sampling.top_p = req.top_p;
        sampling.seed = req.seed;
        if let Some(eos) = tokenizer.eos_id() {
            sampling.stop_token_ids.push(eos);
        }
        Ok(PreparedRequest {
            model: req.model.clone(),
            conversation,
            prompt: rendered.prompt,
            input_ids,
            sampling,
            max_tokens: req.max_tokens.unwrap_or(256).max(1),
        })
    }

    pub fn complete(&self, prepared: &PreparedRequest, tokenizer: &Tokenizer, full_ids: &[u32]) -> Result<ChatCompletionResponse> {
        let prompt_len = prepared.input_ids.len();
        let new_ids = if full_ids.len() > prompt_len { &full_ids[prompt_len..] } else { &[] };
        let content = tokenizer.decode(new_ids)?;
        let finish = if new_ids.last().map(|t| prepared.sampling.stop_token_ids.contains(t)).unwrap_or(false) {
            "stop"
        } else {
            "length"
        };
        Ok(ChatCompletionResponse::from_assistant(
            &prepared.model,
            content.trim().to_string(),
            prompt_len,
            new_ids.len(),
            finish,
        ))
    }
}
