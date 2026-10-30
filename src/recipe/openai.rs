use serde::{Deserialize, Serialize};
use super::types::{Conversation, Message, Role};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole { System, User, Assistant, Tool }

impl From<ChatRole> for Role {
    fn from(r: ChatRole) -> Self {
        match r {
            ChatRole::System => Role::System,
            ChatRole::User => Role::User,
            ChatRole::Assistant => Role::Assistant,
            ChatRole::Tool => Role::Tool,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub stop: Option<Stop>,
    #[serde(default)]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Stop { One(String), Many(Vec<String>) }

impl ChatCompletionRequest {
    pub fn to_conversation(&self) -> Conversation {
        Conversation::from_messages(
            self.messages.iter().map(|m| Message {
                role: m.role.clone().into(),
                content: m.content.clone(),
                name: m.name.clone(),
            }).collect(),
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionMessage { pub role: String, pub content: String }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatCompletionMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: Usage,
}

impl ChatCompletionResponse {
    pub fn from_assistant(model: &str, content: String, prompt_tokens: usize, completion_tokens: usize, finish_reason: &str) -> Self {
        let created = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        Self {
            id: format!("chatcmpl-sentinel-{}", created),
            object: "chat.completion".into(),
            created,
            model: model.to_string(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatCompletionMessage { role: "assistant".into(), content },
                finish_reason: finish_reason.into(),
            }],
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
            },
        }
    }
}
