//! Chat/query request types accepted by [`SippEngine`](super::SippEngine).
//!
//! All types here are public API: builders for one-shot prompts and chat
//! conversations, the `QueryOptions` knobs shared by both, and the internal
//! runtime enqueue path for those requests.

use serde_json::json;

pub use crate::core::{ChatMessage, ChatRole};
use futures_channel::mpsc;

use crate::engine::protocol::{EmbedRequest, EngineEvent};
use crate::engine::{GenerateOptions, RequestSampling, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS};
use crate::error::{Error, Result};
use crate::runtime::InferenceRuntime;

use super::events::emit_event;
use super::token_emission::{start_engine_token_emission, ActiveTokenEmission};
use super::{runtime_command, EngineEventSubscribers};

const MAX_TOKENS_POSITIVE: &str = "max_tokens must be positive";
const CHAT_MESSAGES_REQUIRED: &str = "chat messages must not be empty";
const EMPTY_CHAT_TEMPLATE_PROMPT: &str = "model chat template did not produce a prompt";

#[derive(Debug, Clone, PartialEq)]
pub struct QueryOptions {
    pub context_key: String,
    pub max_tokens: i32,
    pub grammar: String,
    pub json_schema: String,
    pub stop: Vec<String>,
    pub sampling: Option<RequestSampling>,
    pub media: Vec<Vec<u8>>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            context_key: DEFAULT_CONTEXT_KEY.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
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
            context_key: options.cache_key.unwrap_or(DEFAULT_CONTEXT_KEY.to_string()),
            max_tokens: options.max_tokens,
            grammar: options.grammar.unwrap_or_default(),
            json_schema: options.json_schema.unwrap_or_default(),
            stop: options.stop,
            sampling: options.sampling.map(RequestSampling::Full),
            media: Vec::new(),
        }
    }
}

pub struct QueryRequest {
    pub prompt: String,
    pub options: QueryOptions,
    pub emit_tokens: bool,
}

impl QueryRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            options: QueryOptions::default(),
            emit_tokens: false,
        }
    }

    pub fn options(mut self, options: QueryOptions) -> Self {
        self.options = options;
        self
    }

    pub fn emit_tokens(mut self, emit_tokens: bool) -> Self {
        self.emit_tokens = emit_tokens;
        self
    }
}

pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub options: QueryOptions,
    pub emit_tokens: bool,
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            options: QueryOptions::default(),
            emit_tokens: false,
        }
    }

    pub fn options(mut self, options: QueryOptions) -> Self {
        self.options = options;
        self
    }

    pub fn emit_tokens(mut self, emit_tokens: bool) -> Self {
        self.emit_tokens = emit_tokens;
        self
    }
}

pub(super) fn start_embed(
    runtime: &mut InferenceRuntime,
    request: EmbedRequest,
    event_subscribers: &EngineEventSubscribers,
) -> Result<(u32, Option<ActiveTokenEmission>)> {
    let EmbedRequest { input, options } = request;
    let request_id = runtime.enqueue_embed_request(input, options)?;
    emit_request_started(event_subscribers, request_id);
    // Embedding requests never stream tokens.
    Ok((request_id, None))
}

pub(super) fn start_chat(
    runtime: &mut InferenceRuntime,
    request: ChatRequest,
    token_tx: Option<mpsc::UnboundedSender<crate::core::TokenBatch>>,
    event_subscribers: &EngineEventSubscribers,
) -> Result<(u32, Option<ActiveTokenEmission>)> {
    if !runtime.capabilities().has_chat_template {
        return Err(Error::UnsupportedOperation {
            operation: "chat",
            reason: "loaded model has no chat template; call query() with a raw \
                     prompt instead"
                .to_string(),
        });
    }
    let prompt = render_chat_prompt(runtime, &request.messages)?;
    start_query(
        runtime,
        QueryRequest {
            prompt,
            options: request.options,
            emit_tokens: request.emit_tokens,
        },
        token_tx,
        event_subscribers,
    )
}

pub(super) fn start_query(
    runtime: &mut InferenceRuntime,
    request: QueryRequest,
    token_tx: Option<mpsc::UnboundedSender<crate::core::TokenBatch>>,
    event_subscribers: &EngineEventSubscribers,
) -> Result<(u32, Option<ActiveTokenEmission>)> {
    let QueryRequest {
        prompt,
        options,
        emit_tokens,
    } = request;

    if options.max_tokens <= 0 {
        return Err(Error::InvalidRequest(MAX_TOKENS_POSITIVE));
    }

    let should_emit_tokens = emit_tokens && token_tx.is_some();

    let request_id = if options.media.is_empty() {
        runtime.enqueue_request(
            options.context_key,
            prompt,
            options.max_tokens,
            options.grammar,
            options.json_schema,
            options.stop,
            options.sampling,
            should_emit_tokens,
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
            should_emit_tokens,
        )?
    };

    emit_request_started(event_subscribers, request_id);

    Ok((
        request_id,
        attach_token_emission(runtime, request_id, token_tx),
    ))
}

fn emit_request_started(event_subscribers: &EngineEventSubscribers, request_id: u32) {
    emit_event(
        event_subscribers,
        EngineEvent::RequestStarted {
            request_id: request_id.to_string(),
            stream_id: request_id,
        },
    );
}

fn attach_token_emission(
    runtime: &mut InferenceRuntime,
    request_id: u32,
    token_tx: Option<mpsc::UnboundedSender<crate::core::TokenBatch>>,
) -> Option<ActiveTokenEmission> {
    let token = token_tx.map(|token_tx| start_engine_token_emission(request_id, token_tx));
    if let Some(token) = &token {
        runtime
            .request_queue
            .token_emission_sinks
            .insert(request_id, token.producer.clone());
    }
    token
}

fn render_chat_prompt(runtime: &InferenceRuntime, messages: &[ChatMessage]) -> Result<String> {
    if messages.is_empty() {
        return Err(Error::InvalidRequest(CHAT_MESSAGES_REQUIRED));
    }
    let messages_json = render_messages_json(messages)?;
    let prompt = runtime.apply_chat_template_json(&messages_json, true)?;
    if prompt.is_empty() {
        return Err(runtime_command(EMPTY_CHAT_TEMPLATE_PROMPT));
    }
    Ok(prompt)
}

fn render_messages_json(messages: &[ChatMessage]) -> Result<String> {
    let rendered: Vec<_> = messages
        .iter()
        .map(|message| {
            json!({
                "role": message.role.as_str(),
                "content": message.content,
            })
        })
        .collect();
    serde_json::to_string(&rendered)
        .map_err(|error| runtime_command(format!("failed to render chat JSON: {error}")))
}

#[cfg(test)]
#[path = "../../tests/engine/driver/request_tests.rs"]
mod request_tests;
