//! Chat/query request types accepted by [`CogentEngine`](super::CogentEngine).
//!
//! All types here are public API: builders for one-shot prompts and chat
//! conversations, the `QueryOptions` knobs shared by both, and the internal
//! runtime enqueue path for those requests.

use serde_json::json;

use crate::engine::protocol::EngineEvent;
use crate::engine::{stream::TokenBatch, GenerateOptions, SamplingRuntimeConfig};
use crate::error::{Error, Result};
use crate::runtime::request::GenerateTokenEmissionMode;
use crate::runtime::InferenceRuntime;

use super::events::emit_event;
use super::token_sink::{start_async_token_sink, AsyncTokenSink};
use super::{EngineEventSubscribers, OnTokensCallback};

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

pub(super) fn start_chat(
    runtime: &mut InferenceRuntime,
    request: ChatRequest,
    event_subscribers: &EngineEventSubscribers,
) -> Result<(u32, Option<AsyncTokenSink>)> {
    let prompt = render_chat_prompt(runtime, &request.messages)?;
    start_query(
        runtime,
        QueryRequest {
            prompt,
            options: request.options,
            on_tokens: request.on_tokens,
        },
        event_subscribers,
    )
}

pub(super) fn start_query(
    runtime: &mut InferenceRuntime,
    request: QueryRequest,
    event_subscribers: &EngineEventSubscribers,
) -> Result<(u32, Option<AsyncTokenSink>)> {
    let QueryRequest {
        prompt,
        options,
        on_tokens,
    } = request;

    if options.max_tokens <= 0 {
        return Err(Error::InvalidRequest("max_tokens must be positive"));
    }

    let emit_tokens = on_tokens.is_some();
    let emission_mode = if emit_tokens {
        GenerateTokenEmissionMode::TokenStream
    } else {
        GenerateTokenEmissionMode::None
    };

    let request_id = if options.media.is_empty() {
        runtime.enqueue_request(
            options.context_key,
            prompt,
            options.max_tokens,
            options.grammar,
            options.json_schema,
            options.stop,
            options.sampling,
            emission_mode,
        )?
    } else {
        runtime.enqueue_multimodal_request(
            options.context_key,
            prompt,
            options.max_tokens,
            options.media,
            options.grammar,
            options.json_schema,
            options.stop,
            options.sampling,
            emission_mode,
        )?
    };

    emit_event(
        event_subscribers,
        EngineEvent::RequestStarted {
            request_id: request_id.to_string(),
            stream_id: request_id,
        },
    );

    let token_sink = on_tokens.map(|callback| start_async_token_sink(request_id, callback));
    if let Some(sink) = &token_sink {
        runtime.add_token_ring_producer(request_id, sink.producer.clone());
    }

    Ok((request_id, token_sink))
}

fn render_chat_prompt(runtime: &InferenceRuntime, messages: &[ChatMessage]) -> Result<String> {
    if messages.is_empty() {
        return Err(Error::InvalidRequest("chat messages must not be empty"));
    }
    let messages_json = render_messages_json(messages)?;
    let prompt = runtime.apply_chat_template_json(&messages_json, true)?;
    if prompt.is_empty() {
        return Err(Error::RuntimeCommand(
            "model chat template did not produce a prompt".to_string(),
        ));
    }
    Ok(prompt)
}

fn render_messages_json(messages: &[ChatMessage]) -> Result<String> {
    let mut rendered = Vec::with_capacity(messages.len());
    rendered.extend(messages.iter().map(|message| {
        json!({
            "role": message.role.as_str(),
            "content": message.content,
        })
    }));
    serde_json::to_string(&rendered)
        .map_err(|error| Error::RuntimeCommand(format!("failed to render chat JSON: {error}")))
}

#[cfg(test)]
mod tests {
    use super::{render_messages_json, ChatMessage};

    #[test]
    fn render_messages_json_preserves_role_and_content_order() {
        let messages = [
            ChatMessage::system("policy"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ];

        let json = render_messages_json(&messages).expect("messages json");

        assert_eq!(
            json,
            r#"[{"content":"policy","role":"system"},{"content":"hello","role":"user"},{"content":"hi","role":"assistant"}]"#
        );
    }
}
