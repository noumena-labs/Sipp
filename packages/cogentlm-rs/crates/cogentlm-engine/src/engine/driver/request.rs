//! Chat/query request types accepted by [`CogentEngine`](super::CogentEngine).
//!
//! All types here are public API: builders for one-shot prompts and chat
//! conversations, the `QueryOptions` knobs shared by both, and the internal
//! runtime enqueue path for those requests.

use serde_json::json;

pub use cogentlm_core::{ChatMessage, ChatRole};

use crate::engine::protocol::{EmbedRequest, EngineEvent};
use crate::engine::{
    stream::TokenBatch, GenerateOptions, SamplingRuntimeConfig, DEFAULT_CONTEXT_KEY,
    DEFAULT_MAX_TOKENS,
};
use crate::error::{Error, Result};
use crate::runtime::request::GenerateTokenEmissionMode;
use crate::runtime::InferenceRuntime;

use super::events::emit_event;
use super::token_sink::{start_async_token_sink, AsyncTokenSink};
use super::{runtime_command, EngineEventSubscribers, OnTokensCallback};

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
    pub sampling: Option<SamplingRuntimeConfig>,
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

pub(super) fn start_embed(
    runtime: &mut InferenceRuntime,
    request: EmbedRequest,
    event_subscribers: &EngineEventSubscribers,
) -> Result<(u32, Option<AsyncTokenSink>)> {
    let EmbedRequest { input, options } = request;
    let request_id = runtime.enqueue_embed_request(input, options)?;
    emit_request_started(event_subscribers, request_id);
    // Embedding requests never stream tokens; there is no on_tokens hook.
    Ok((request_id, None))
}

pub(super) fn start_chat(
    runtime: &mut InferenceRuntime,
    request: ChatRequest,
    event_subscribers: &EngineEventSubscribers,
) -> Result<(u32, Option<AsyncTokenSink>)> {
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
        return Err(Error::InvalidRequest(MAX_TOKENS_POSITIVE));
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

    emit_request_started(event_subscribers, request_id);

    Ok((
        request_id,
        attach_token_sink(runtime, request_id, on_tokens),
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

fn attach_token_sink(
    runtime: &mut InferenceRuntime,
    request_id: u32,
    on_tokens: Option<OnTokensCallback>,
) -> Option<AsyncTokenSink> {
    let token_sink = on_tokens.map(|callback| start_async_token_sink(request_id, callback));
    if let Some(sink) = &token_sink {
        runtime
            .request_queue
            .token_ring_producers
            .insert(request_id, sink.producer.clone());
    }
    token_sink
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
mod tests {
    mod request_tests;
}
