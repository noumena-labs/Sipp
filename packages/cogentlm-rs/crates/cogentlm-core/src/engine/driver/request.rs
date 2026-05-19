//! Chat/query request types accepted by [`CogentEngine`](super::CogentEngine).
//!
//! All types here are public API: builders for one-shot prompts and chat
//! conversations, plus the `QueryOptions` knobs shared by both.

use crate::engine::{stream::TokenBatch, GenerateOptions, SamplingRuntimeConfig};
use crate::error::Result;

use super::OnTokensCallback;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryOptions {
    pub context_key: String,
    pub max_tokens: i32,
    pub grammar: String,
    pub json_schema: String,
    pub stop: Vec<String>,
    pub sampling: Option<SamplingRuntimeConfig>,
    pub media: Vec<Vec<u8>>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            context_key: "default".to_string(),
            max_tokens: 64,
            grammar: String::new(),
            json_schema: String::new(),
            stop: Vec::new(),
            sampling: None,
            media: Vec::new(),
        }
    }
}

impl From<GenerateOptions> for QueryOptions {
    fn from(options: GenerateOptions) -> Self {
        Self {
            context_key: options.cache_key.unwrap_or_else(|| "default".to_string()),
            max_tokens: options.max_tokens,
            grammar: options.grammar.unwrap_or_default(),
            json_schema: options.json_schema.unwrap_or_default(),
            stop: options.stop,
            sampling: options.sampling,
            media: Vec::new(),
        }
    }
}

pub struct QueryRequest {
    pub prompt: String,
    pub options: QueryOptions,
    pub(super) on_tokens: Option<OnTokensCallback>,
}

impl QueryRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            options: QueryOptions::default(),
            on_tokens: None,
        }
    }

    pub fn options(mut self, options: QueryOptions) -> Self {
        self.options = options;
        self
    }

    pub fn on_tokens(
        mut self,
        callback: impl FnMut(&TokenBatch) -> Result<()> + Send + 'static,
    ) -> Self {
        self.on_tokens = Some(Box::new(callback));
        self
    }
}

impl From<String> for QueryRequest {
    fn from(prompt: String) -> Self {
        Self::new(prompt)
    }
}

impl From<&str> for QueryRequest {
    fn from(prompt: &str) -> Self {
        Self::new(prompt)
    }
}

pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub options: QueryOptions,
    pub(super) on_tokens: Option<OnTokensCallback>,
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            options: QueryOptions::default(),
            on_tokens: None,
        }
    }

    pub fn options(mut self, options: QueryOptions) -> Self {
        self.options = options;
        self
    }

    pub fn on_tokens(
        mut self,
        callback: impl FnMut(&TokenBatch) -> Result<()> + Send + 'static,
    ) -> Self {
        self.on_tokens = Some(Box::new(callback));
        self
    }
}

impl From<Vec<ChatMessage>> for ChatRequest {
    fn from(messages: Vec<ChatMessage>) -> Self {
        Self::new(messages)
    }
}
