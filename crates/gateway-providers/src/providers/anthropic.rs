use std::collections::VecDeque;

use async_trait::async_trait;
use cogentlm_core::{ChatMessage, ChatRole, FinishReason};
use futures_util::{stream as futures_stream, StreamExt};

use crate::error::provider_error_kind_from_code;
use crate::stream::{SseParser, TokenBatchBuilder};
use crate::{
    AnthropicAdapterConfig, CapabilitySupport, GatewayBackendAdapter, HttpByteStream,
    HttpTransport, ProviderAuth, ProviderCapabilities, ProviderChatRequest, ProviderChatResponse,
    ProviderEmbedRequest, ProviderEmbeddingResponse, ProviderError, ProviderErrorKind,
    ProviderGenerateRequest, ProviderGenerateResponse, ProviderKind, ProviderModel,
    ProviderResponse, ProviderResponseMetadata, ProviderResult, ProviderStream,
    ProviderStreamEvent, ProviderTextOutput, TokenUsage,
};

use super::common::{
    insert_finite_f32_option, insert_positive_u32_option, merge_provider_options, optional_u32,
    provider_body_error, provider_response_error, require_non_empty_field, token_usage_total,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "../tests/providers/anthropic_tests.rs"]
mod anthropic_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
const DEFAULT_ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_ANTHROPIC_MAX_TOKENS: u32 = 1024;
const ANTHROPIC_CHAT_TYPED_FIELDS: &[&str] = &[
    "model",
    "max_tokens",
    "messages",
    "system",
    "temperature",
    "top_p",
    "stop_sequences",
    "stream",
];

pub struct AnthropicAdapter {
    transport: HttpTransport,
}

impl AnthropicAdapter {
    pub fn new(config: AnthropicAdapterConfig) -> ProviderResult<Self> {
        let base_url = config
            .base_url
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_BASE_URL.to_string());
        let version = config
            .version
            .unwrap_or_else(|| DEFAULT_ANTHROPIC_VERSION.to_string());
        require_non_empty_field(&version, "anthropic-version", ProviderKind::Anthropic)?;

        let transport = HttpTransport::new_with_options(
            ProviderKind::Anthropic,
            base_url,
            ProviderAuth::Header {
                name: "x-api-key".to_string(),
                value: config.api_key,
            },
            vec![("anthropic-version".to_string(), version)],
            config.timeout,
        )?;
        Ok(Self { transport })
    }
}

#[async_trait]
impl GatewayBackendAdapter for AnthropicAdapter {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Anthropic
    }

    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>> {
        let response = self.transport.get_json("/models").await?;
        anthropic_models_from_body(&response.body)
    }

    async fn get_model(&self, model: &str) -> ProviderResult<ProviderModel> {
        let response = self.transport.get_json(&format!("/models/{model}")).await?;
        anthropic_model_from_value(&response.body)
    }

    async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse> {
        let body = anthropic_chat_body(&req, false)?;
        let response = self.transport.post_json("/messages", &body).await?;
        anthropic_text_response(response.request_id, response.body)
    }

    async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        let body = anthropic_chat_body(&anthropic_generate_chat_request(req), false)?;
        let response = self.transport.post_json("/messages", &body).await?;
        anthropic_text_response(response.request_id, response.body)
    }

    async fn stream_generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        let body = anthropic_chat_body(&anthropic_generate_chat_request(req), true)?;
        let response = self.transport.post_json_stream("/messages", &body).await?;
        Ok(anthropic_stream_events(
            response.request_id,
            response.stream,
        ))
    }

    async fn embed(&self, _req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse> {
        Err(ProviderError::new(
            ProviderErrorKind::UnsupportedFeature,
            ProviderKind::Anthropic,
            "Anthropic native provider does not expose embeddings",
        ))
    }

    async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        let body = anthropic_chat_body(&req, true)?;
        let response = self.transport.post_json_stream("/messages", &body).await?;
        Ok(anthropic_stream_events(
            response.request_id,
            response.stream,
        ))
    }
}

fn anthropic_generate_chat_request(req: ProviderGenerateRequest) -> ProviderChatRequest {
    ProviderChatRequest {
        model: req.model,
        messages: vec![ChatMessage::new(ChatRole::User, req.prompt)],
        options: req.options,
        provider_options: req.provider_options,
    }
}

fn anthropic_chat_body(
    req: &ProviderChatRequest,
    stream: bool,
) -> ProviderResult<serde_json::Value> {
    require_non_empty_field(&req.model, "model", ProviderKind::Anthropic)?;

    let (system, messages) = anthropic_messages(&req.messages);
    if messages.is_empty() {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            ProviderKind::Anthropic,
            "Anthropic messages must include at least one user or assistant message",
        ));
    }

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::Value::String(req.model.clone()),
    );
    body.insert("messages".to_string(), serde_json::Value::Array(messages));
    if let Some(system) = system {
        body.insert("system".to_string(), serde_json::Value::String(system));
    }
    insert_positive_u32_option(
        &mut body,
        "max_tokens",
        Some(
            req.options
                .max_tokens
                .unwrap_or(DEFAULT_ANTHROPIC_MAX_TOKENS),
        ),
        ProviderKind::Anthropic,
    )?;
    insert_finite_f32_option(
        &mut body,
        "temperature",
        req.options.temperature,
        ProviderKind::Anthropic,
    )?;
    insert_finite_f32_option(
        &mut body,
        "top_p",
        req.options.top_p,
        ProviderKind::Anthropic,
    )?;
    if !req.options.stop.is_empty() {
        body.insert(
            "stop_sequences".to_string(),
            serde_json::json!(req.options.stop),
        );
    }
    if stream {
        body.insert("stream".to_string(), serde_json::json!(true));
    }

    merge_provider_options(
        &mut body,
        &req.provider_options,
        ANTHROPIC_CHAT_TYPED_FIELDS,
        ProviderKind::Anthropic,
    )?;
    Ok(serde_json::Value::Object(body))
}

fn anthropic_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system = Vec::new();
    let mut conversation = Vec::new();

    for message in messages {
        let role = match message.role {
            ChatRole::System => {
                if !message.content.trim().is_empty() {
                    system.push(message.content.clone());
                }
                continue;
            }
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        conversation.push(serde_json::json!({
            "role": role,
            "content": message.content,
        }));
    }

    let system = (!system.is_empty()).then(|| system.join("\n\n"));
    (system, conversation)
}

fn anthropic_models_from_body(body: &serde_json::Value) -> ProviderResult<Vec<ProviderModel>> {
    let data = body
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            provider_response_error(
                "Anthropic models response missing data array",
                ProviderKind::Anthropic,
            )
        })?;
    data.iter().map(anthropic_model_from_value).collect()
}

fn anthropic_model_from_value(value: &serde_json::Value) -> ProviderResult<ProviderModel> {
    let id = value
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            provider_response_error(
                "Anthropic model response missing id",
                ProviderKind::Anthropic,
            )
        })?;

    Ok(ProviderModel {
        id: id.to_string(),
        provider: ProviderKind::Anthropic,
        display_name: value
            .get("display_name")
            .and_then(serde_json::Value::as_str)
            .map(str::to_owned),
        capabilities: ProviderCapabilities {
            chat: CapabilitySupport::Supported,
            generate: CapabilitySupport::Supported,
            embeddings: CapabilitySupport::Unsupported,
            token_emission: CapabilitySupport::Supported,
        },
        context_window: None,
        max_output_tokens: None,
        raw: value.clone(),
    })
}

fn anthropic_text_response(
    request_id: Option<String>,
    body: serde_json::Value,
) -> ProviderResult<ProviderResponse<ProviderTextOutput>> {
    if body.get("error").is_some_and(|value| !value.is_null()) {
        return Err(provider_body_error(
            body,
            ProviderKind::Anthropic,
            "Anthropic response error",
        ));
    }

    let text = anthropic_text_from_content(&body)?;
    let finish_reason_raw = body
        .get("stop_reason")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let finish_reason = anthropic_finish_reason(finish_reason_raw.as_deref());
    let response_model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            provider_response_error("Anthropic response missing model", ProviderKind::Anthropic)
        })?
        .to_string();
    let response_id = body
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let usage = body
        .get("usage")
        .filter(|value| !value.is_null())
        .map(anthropic_usage)
        .transpose()?;

    Ok(ProviderResponse {
        result: ProviderTextOutput {
            text,
            finish_reason,
        },
        usage,
        metadata: ProviderResponseMetadata {
            provider: ProviderKind::Anthropic,
            model: response_model,
            request_id,
            response_id,
            finish_reason_raw,
            raw: body,
        },
    })
}

fn anthropic_text_from_content(body: &serde_json::Value) -> ProviderResult<String> {
    let content = body
        .get("content")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            provider_response_error(
                "Anthropic response missing content array",
                ProviderKind::Anthropic,
            )
        })?;
    let mut text = String::new();
    for block in content {
        if block.get("type").and_then(serde_json::Value::as_str) != Some("text") {
            continue;
        }
        let block_text = block
            .get("text")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                provider_response_error(
                    "Anthropic text content block missing text",
                    ProviderKind::Anthropic,
                )
            })?;
        text.push_str(block_text);
    }

    Ok(text)
}

fn anthropic_finish_reason(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("max_tokens") => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

fn anthropic_usage(value: &serde_json::Value) -> ProviderResult<TokenUsage> {
    let input_tokens = checked_usage_sum(
        [
            optional_u32(value, "input_tokens", ProviderKind::Anthropic)?,
            optional_u32(
                value,
                "cache_creation_input_tokens",
                ProviderKind::Anthropic,
            )?,
            optional_u32(value, "cache_read_input_tokens", ProviderKind::Anthropic)?,
        ],
        "input_tokens",
    )?;
    let output_tokens = optional_u32(value, "output_tokens", ProviderKind::Anthropic)?;

    Ok(TokenUsage {
        input_tokens,
        output_tokens,
        total_tokens: token_usage_total(input_tokens, output_tokens),
    })
}

fn checked_usage_sum(values: [Option<u32>; 3], field: &'static str) -> ProviderResult<Option<u32>> {
    let mut total: Option<u32> = None;
    for value in values.into_iter().flatten() {
        total = Some(match total {
            Some(total) => total.checked_add(value).ok_or_else(|| {
                provider_response_error(
                    format!("usage field exceeds u32: {field}"),
                    ProviderKind::Anthropic,
                )
            })?,
            None => value,
        });
    }
    Ok(total)
}

struct AnthropicStreamState {
    request_id: Option<String>,
    stream: HttpByteStream,
    parser: SseParser,
    pending: VecDeque<ProviderResult<ProviderStreamEvent>>,
    batcher: TokenBatchBuilder,
    usage: TokenUsage,
    closed: bool,
    finished: bool,
    finish_reason: Option<FinishReason>,
    missing_stop_reported: bool,
}

fn anthropic_stream_events(
    request_id: Option<String>,
    byte_stream: HttpByteStream,
) -> ProviderStream<ProviderStreamEvent> {
    let state = AnthropicStreamState {
        request_id: request_id.clone(),
        stream: byte_stream,
        parser: SseParser::new(ProviderKind::Anthropic),
        pending: VecDeque::new(),
        batcher: TokenBatchBuilder::new(request_id),
        usage: TokenUsage {
            input_tokens: None,
            output_tokens: None,
            total_tokens: None,
        },
        closed: false,
        finished: false,
        finish_reason: None,
        missing_stop_reported: false,
    };

    Box::pin(futures_stream::unfold(state, next_anthropic_stream_event))
}

async fn next_anthropic_stream_event(
    mut state: AnthropicStreamState,
) -> Option<(ProviderResult<ProviderStreamEvent>, AnthropicStreamState)> {
    loop {
        if let Some(event) = state.pending.pop_front() {
            return Some((event, state));
        }
        if state.closed {
            if !state.finished && !state.missing_stop_reported {
                state.missing_stop_reported = true;
                return Some((Err(state.missing_stop_error()), state));
            }
            return None;
        }

        match state.stream.next().await {
            Some(Ok(bytes)) => {
                if let Err(err) = state.push_bytes(&bytes) {
                    state.closed = true;
                    state.missing_stop_reported = true;
                    state.pending.clear();
                    return Some((Err(err), state));
                }
            }
            Some(Err(err)) => {
                state.closed = true;
                state.missing_stop_reported = true;
                state.pending.clear();
                return Some((Err(err), state));
            }
            None => {
                state.closed = true;
                if let Err(err) = state.finish_parser() {
                    state.missing_stop_reported = true;
                    state.pending.clear();
                    return Some((Err(err), state));
                }
            }
        }
    }
}

impl AnthropicStreamState {
    fn push_bytes(&mut self, bytes: &[u8]) -> ProviderResult<()> {
        let payloads = self
            .parser
            .push(bytes)
            .map_err(|err| self.with_request_id(err))?;
        for payload in payloads {
            self.push_payload(&payload)
                .map_err(|err| self.with_request_id(err))?;
        }
        Ok(())
    }

    fn finish_parser(&mut self) -> ProviderResult<()> {
        let payloads = self
            .parser
            .finish()
            .map_err(|err| self.with_request_id(err))?;
        for payload in payloads {
            self.push_payload(&payload)
                .map_err(|err| self.with_request_id(err))?;
        }
        Ok(())
    }

    fn push_payload(&mut self, payload: &str) -> ProviderResult<()> {
        let value = serde_json::from_str::<serde_json::Value>(payload).map_err(|err| {
            provider_response_error(
                format!("invalid Anthropic SSE JSON payload: {err}"),
                ProviderKind::Anthropic,
            )
        })?;
        if !value.is_object() {
            return Err(provider_response_error(
                "Anthropic stream payload must be a JSON object",
                ProviderKind::Anthropic,
            ));
        }
        if value.get("type").and_then(serde_json::Value::as_str) == Some("error")
            || value.get("error").is_some_and(|value| !value.is_null())
        {
            return Err(self.with_request_id(anthropic_stream_error(value)));
        }

        let event_type = value.get("type").and_then(serde_json::Value::as_str);
        if self.finished {
            if event_type == Some("message_stop") {
                return Ok(());
            }
            return Err(self.event_after_stop_error());
        }

        match event_type {
            Some("message_start") => self.push_message_start(&value),
            Some("content_block_delta") => self.push_content_block_delta(&value),
            Some("message_delta") => self.push_message_delta(&value),
            Some("message_stop") => {
                if !self.finished {
                    self.push_finished(self.finish_reason.unwrap_or(FinishReason::Stop));
                }
                Ok(())
            }
            Some("ping" | "content_block_start" | "content_block_stop") | None => Ok(()),
            Some(_) => Ok(()),
        }
    }

    fn push_message_start(&mut self, value: &serde_json::Value) -> ProviderResult<()> {
        if let Some(usage) = value
            .pointer("/message/usage")
            .filter(|value| !value.is_null())
        {
            self.push_usage(anthropic_usage(usage)?);
        }
        Ok(())
    }

    fn push_content_block_delta(&mut self, value: &serde_json::Value) -> ProviderResult<()> {
        let Some(delta) = value.get("delta") else {
            return Ok(());
        };
        if delta.get("type").and_then(serde_json::Value::as_str) == Some("text_delta") {
            let text = delta
                .get("text")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| {
                    provider_response_error(
                        "Anthropic text_delta missing text",
                        ProviderKind::Anthropic,
                    )
                })?;
            self.push_text(text);
        }
        Ok(())
    }

    fn push_message_delta(&mut self, value: &serde_json::Value) -> ProviderResult<()> {
        if let Some(usage) = value.get("usage").filter(|value| !value.is_null()) {
            self.push_usage(anthropic_usage(usage)?);
        }
        if let Some(raw) = value
            .pointer("/delta/stop_reason")
            .and_then(serde_json::Value::as_str)
        {
            self.finish_reason = Some(anthropic_finish_reason(Some(raw)));
        }
        Ok(())
    }

    fn push_text(&mut self, text: &str) {
        let batch = self.batcher.push_text(text);
        self.pending
            .push_back(Ok(ProviderStreamEvent::TokenBatch(batch)));
    }

    fn push_usage(&mut self, usage: TokenUsage) {
        if let Some(input_tokens) = usage.input_tokens {
            self.usage.input_tokens = Some(input_tokens);
        }
        if let Some(output_tokens) = usage.output_tokens {
            self.usage.output_tokens = Some(output_tokens);
        }
        self.usage.total_tokens =
            token_usage_total(self.usage.input_tokens, self.usage.output_tokens)
                .or(usage.total_tokens);

        self.pending
            .push_back(Ok(ProviderStreamEvent::Usage { usage: self.usage }));
    }

    fn push_finished(&mut self, finish_reason: FinishReason) {
        if self.finished {
            return;
        }
        self.finished = true;
        self.pending
            .push_back(Ok(ProviderStreamEvent::Finished { finish_reason }));
    }

    fn with_request_id(&self, mut err: ProviderError) -> ProviderError {
        if err.request_id.is_none() {
            err.request_id = self.request_id.clone();
        }
        err
    }

    fn missing_stop_error(&self) -> ProviderError {
        self.with_request_id(provider_response_error(
            "Anthropic stream ended before message_stop",
            ProviderKind::Anthropic,
        ))
    }

    fn event_after_stop_error(&self) -> ProviderError {
        self.with_request_id(provider_response_error(
            "Anthropic stream event received after message_stop",
            ProviderKind::Anthropic,
        ))
    }
}

fn anthropic_stream_error(raw: serde_json::Value) -> ProviderError {
    let message = raw
        .pointer("/error/message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("Anthropic stream error")
        .to_string();
    let code = raw
        .pointer("/error/type")
        .or_else(|| raw.pointer("/error/code"))
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);

    ProviderError {
        kind: provider_error_kind_from_code(code.as_deref()).unwrap_or(ProviderErrorKind::Provider),
        provider: ProviderKind::Anthropic,
        status: None,
        code,
        message,
        retry_after: None,
        request_id: None,
        raw: Some(Box::new(raw)),
    }
}
