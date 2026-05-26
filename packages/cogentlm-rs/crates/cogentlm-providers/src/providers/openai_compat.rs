//! OpenAI-compatible wire format shared by the proxy and direct OpenAI backends.
//! Request/response translation, usage/finish mapping, and SSE stream parsing
//! live here so backends that speak the OpenAI protocol reuse one implementation.

use std::collections::VecDeque;

use cogentlm_core::{ChatMessage, ChatRole, FinishReason};
use futures_util::{stream as futures_stream, StreamExt};

use crate::error::provider_error_kind_from_code;
use crate::stream::{SseParser, TokenBatchBuilder};
use crate::{
    CapabilitySupport, HttpByteStream, ProviderCapabilities, ProviderChatRequest,
    ProviderChatResponse, ProviderEmbedRequest, ProviderEmbeddingOutput, ProviderEmbeddingResponse,
    ProviderError, ProviderErrorKind, ProviderKind, ProviderModel, ProviderResponseMetadata,
    ProviderResult, ProviderStream, ProviderStreamEvent, ProviderTextOutput, TokenUsage,
};

use super::common::{
    insert_finite_f32_option, insert_positive_u32_option, merge_provider_options, optional_u32,
    provider_body_error, provider_response_error, require_non_empty_field,
};

pub(super) const OPENAI_CHAT_TYPED_FIELDS: &[&str] = &[
    "model",
    "messages",
    "max_tokens",
    "temperature",
    "top_p",
    "stop",
    "stream",
];
pub(super) const OPENAI_EMBED_TYPED_FIELDS: &[&str] = &["model", "input", "encoding_format"];

pub(super) fn openai_models_from_body(
    body: &serde_json::Value,
    provider: ProviderKind,
) -> ProviderResult<Vec<ProviderModel>> {
    let data = body
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| provider_response_error("models response missing data array", provider))?;
    data.iter()
        .map(|value| openai_model_from_value(value, provider))
        .collect()
}

pub(super) fn openai_model_from_value(
    value: &serde_json::Value,
    provider: ProviderKind,
) -> ProviderResult<ProviderModel> {
    let id = value
        .get("id")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| provider_response_error("model response missing id", provider))?;

    Ok(ProviderModel {
        id: id.to_string(),
        provider,
        display_name: None,
        capabilities: ProviderCapabilities {
            chat: CapabilitySupport::Unknown,
            generate: CapabilitySupport::Unknown,
            embeddings: CapabilitySupport::Unknown,
            streaming: CapabilitySupport::Unknown,
        },
        context_window: None,
        max_output_tokens: None,
        raw: value.clone(),
    })
}

pub(super) fn openai_chat_body(
    req: &ProviderChatRequest,
    provider: ProviderKind,
) -> ProviderResult<serde_json::Value> {
    openai_chat_body_with_stream(req, false, provider)
}

pub(super) fn openai_stream_chat_body(
    req: &ProviderChatRequest,
    provider: ProviderKind,
) -> ProviderResult<serde_json::Value> {
    openai_chat_body_with_stream(req, true, provider)
}

pub(super) fn openai_embedding_body(
    req: &ProviderEmbedRequest,
    provider: ProviderKind,
) -> ProviderResult<serde_json::Value> {
    require_non_empty_field(&req.model, "model", provider)?;
    require_non_empty_field(&req.input, "input", provider)?;

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::Value::String(req.model.clone()),
    );
    body.insert(
        "input".to_string(),
        serde_json::Value::String(req.input.clone()),
    );
    body.insert(
        "encoding_format".to_string(),
        serde_json::Value::String("float".to_string()),
    );

    merge_provider_options(
        &mut body,
        &req.provider_options,
        OPENAI_EMBED_TYPED_FIELDS,
        provider,
    )?;
    Ok(serde_json::Value::Object(body))
}

fn openai_chat_body_with_stream(
    req: &ProviderChatRequest,
    stream: bool,
    provider: ProviderKind,
) -> ProviderResult<serde_json::Value> {
    require_non_empty_field(&req.model, "model", provider)?;
    if req.messages.is_empty() {
        return Err(ProviderError::new(
            ProviderErrorKind::InvalidRequest,
            provider,
            "provider chat messages must not be empty",
        ));
    }

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::Value::String(req.model.clone()),
    );
    body.insert(
        "messages".to_string(),
        serde_json::Value::Array(req.messages.iter().map(openai_message).collect()),
    );

    insert_positive_u32_option(&mut body, "max_tokens", req.options.max_tokens, provider)?;
    insert_finite_f32_option(&mut body, "temperature", req.options.temperature, provider)?;
    insert_finite_f32_option(&mut body, "top_p", req.options.top_p, provider)?;
    if !req.options.stop.is_empty() {
        body.insert("stop".to_string(), serde_json::json!(req.options.stop));
    }
    if stream {
        body.insert("stream".to_string(), serde_json::json!(true));
    }

    merge_provider_options(
        &mut body,
        &req.provider_options,
        OPENAI_CHAT_TYPED_FIELDS,
        provider,
    )?;
    Ok(serde_json::Value::Object(body))
}

fn openai_message(message: &ChatMessage) -> serde_json::Value {
    serde_json::json!({
        "role": openai_role(message.role),
        "content": message.content,
    })
}

fn openai_role(role: ChatRole) -> &'static str {
    match role {
        ChatRole::System => "system",
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
    }
}

pub(super) fn openai_chat_response_from_body(
    request_id: Option<String>,
    body: serde_json::Value,
    provider: ProviderKind,
) -> ProviderResult<ProviderChatResponse> {
    let choice = body
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| provider_response_error("chat response missing first choice", provider))?;
    // A tool-call-only turn has `content: null`; surface empty text and leave the
    // tool_calls in `metadata.raw` for caller-owned handling rather than erroring.
    let text = choice
        .pointer("/message/content")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    let finish_reason_raw = choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let finish_reason = map_finish_reason(finish_reason_raw.as_deref());
    let response_model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| provider_response_error("chat response missing model", provider))?
        .to_string();
    let response_id = body
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let usage = body
        .get("usage")
        .filter(|value| !value.is_null())
        .map(|value| openai_chat_usage(value, provider))
        .transpose()?;

    Ok(ProviderChatResponse {
        result: ProviderTextOutput {
            text: text.to_string(),
            finish_reason,
        },
        usage,
        metadata: ProviderResponseMetadata {
            provider,
            model: response_model,
            request_id,
            response_id,
            finish_reason_raw,
            raw: body,
        },
    })
}

pub(super) fn openai_embedding_response_from_body(
    request_id: Option<String>,
    body: serde_json::Value,
    provider: ProviderKind,
) -> ProviderResult<ProviderEmbeddingResponse> {
    if body.get("error").is_some_and(|value| !value.is_null()) {
        return Err(provider_body_error(
            body,
            provider,
            "provider embedding error",
        ));
    }

    let values = embedding_values(&body, provider)?;
    let response_model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| provider_response_error("embedding response missing model", provider))?
        .to_string();
    let usage = body
        .get("usage")
        .filter(|value| !value.is_null())
        .map(|value| openai_embedding_usage(value, provider))
        .transpose()?;

    Ok(ProviderEmbeddingResponse {
        result: ProviderEmbeddingOutput { values },
        usage,
        metadata: ProviderResponseMetadata {
            provider,
            model: response_model,
            request_id,
            response_id: body
                .get("id")
                .and_then(serde_json::Value::as_str)
                .map(str::to_owned),
            finish_reason_raw: None,
            raw: body,
        },
    })
}

fn embedding_values(body: &serde_json::Value, provider: ProviderKind) -> ProviderResult<Vec<f32>> {
    let data = body
        .get("data")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| provider_response_error("embedding response missing data", provider))?;
    let first = data
        .first()
        .ok_or_else(|| provider_response_error("embedding response data is empty", provider))?;
    let values = first
        .get("embedding")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| provider_response_error("embedding response missing vector", provider))?;

    values
        .iter()
        .map(|value| embedding_value(value, provider))
        .collect()
}

fn embedding_value(value: &serde_json::Value, provider: ProviderKind) -> ProviderResult<f32> {
    let Some(value) = value.as_f64() else {
        return Err(provider_response_error(
            "embedding value is not numeric",
            provider,
        ));
    };
    if !value.is_finite() || value < f64::from(f32::MIN) || value > f64::from(f32::MAX) {
        return Err(provider_response_error(
            "embedding value is not representable as f32",
            provider,
        ));
    }
    Ok(value as f32)
}

pub(super) fn openai_embedding_usage(
    value: &serde_json::Value,
    provider: ProviderKind,
) -> ProviderResult<TokenUsage> {
    Ok(TokenUsage {
        input_tokens: optional_u32(value, "prompt_tokens", provider)?,
        output_tokens: None,
        total_tokens: optional_u32(value, "total_tokens", provider)?,
    })
}

pub(super) fn map_finish_reason(raw: Option<&str>) -> FinishReason {
    match raw {
        Some("length" | "max_tokens" | "max_output_tokens") => FinishReason::Length,
        _ => FinishReason::Stop,
    }
}

pub(super) fn openai_chat_usage(
    value: &serde_json::Value,
    provider: ProviderKind,
) -> ProviderResult<TokenUsage> {
    Ok(TokenUsage {
        input_tokens: optional_u32(value, "prompt_tokens", provider)?,
        output_tokens: optional_u32(value, "completion_tokens", provider)?,
        total_tokens: optional_u32(value, "total_tokens", provider)?,
    })
}

struct OpenAiStreamState {
    stream: HttpByteStream,
    parser: SseParser,
    pending: VecDeque<ProviderResult<ProviderStreamEvent>>,
    batcher: TokenBatchBuilder,
    closed: bool,
    provider: ProviderKind,
}

pub(super) fn openai_stream_events(
    request_id: Option<String>,
    byte_stream: HttpByteStream,
    provider: ProviderKind,
) -> ProviderStream<ProviderStreamEvent> {
    let state = OpenAiStreamState {
        stream: byte_stream,
        parser: SseParser::new(provider),
        pending: VecDeque::new(),
        batcher: TokenBatchBuilder::new(request_id),
        closed: false,
        provider,
    };

    Box::pin(futures_stream::unfold(state, next_openai_stream_event))
}

async fn next_openai_stream_event(
    mut state: OpenAiStreamState,
) -> Option<(ProviderResult<ProviderStreamEvent>, OpenAiStreamState)> {
    loop {
        if let Some(event) = state.pending.pop_front() {
            return Some((event, state));
        }
        if state.closed {
            return None;
        }

        match state.stream.next().await {
            Some(Ok(bytes)) => {
                if let Err(err) = state.push_bytes(&bytes) {
                    state.closed = true;
                    return Some((Err(err), state));
                }
            }
            Some(Err(err)) => {
                state.closed = true;
                return Some((Err(err), state));
            }
            None => {
                state.closed = true;
                if let Err(err) = state.finish_parser() {
                    return Some((Err(err), state));
                }
            }
        }
    }
}

impl OpenAiStreamState {
    fn push_bytes(&mut self, bytes: &[u8]) -> ProviderResult<()> {
        for payload in self.parser.push(bytes)? {
            self.push_payload(&payload)?;
        }
        Ok(())
    }

    fn finish_parser(&mut self) -> ProviderResult<()> {
        for payload in self.parser.finish() {
            self.push_payload(&payload)?;
        }
        Ok(())
    }

    fn push_payload(&mut self, payload: &str) -> ProviderResult<()> {
        if payload.trim() == "[DONE]" {
            return Ok(());
        }

        let value = serde_json::from_str::<serde_json::Value>(payload).map_err(|err| {
            provider_response_error(format!("invalid SSE JSON payload: {err}"), self.provider)
        })?;
        if value.get("error").is_some() {
            return Err(provider_stream_error(value, self.provider));
        }

        if let Some(usage_value) = value.get("usage").filter(|value| !value.is_null()) {
            self.pending.push_back(Ok(ProviderStreamEvent::Usage {
                usage: openai_chat_usage(usage_value, self.provider)?,
            }));
        }

        let Some(choices) = value.get("choices").and_then(serde_json::Value::as_array) else {
            return Ok(());
        };
        for choice in choices {
            self.push_choice(choice);
        }
        Ok(())
    }

    fn push_choice(&mut self, choice: &serde_json::Value) {
        if let Some(text) = choice
            .pointer("/delta/content")
            .and_then(serde_json::Value::as_str)
        {
            self.push_text(text);
        }

        if let Some(raw) = choice
            .get("finish_reason")
            .and_then(serde_json::Value::as_str)
        {
            self.pending.push_back(Ok(ProviderStreamEvent::Finished {
                finish_reason: map_finish_reason(Some(raw)),
            }));
        }
    }

    fn push_text(&mut self, text: &str) {
        let batch = self.batcher.push_text(text);
        self.pending
            .push_back(Ok(ProviderStreamEvent::TokenBatch(batch)));
    }
}

fn provider_stream_error(raw: serde_json::Value, provider: ProviderKind) -> ProviderError {
    let message = raw
        .pointer("/error/message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("provider stream error")
        .to_string();
    let code = raw
        .pointer("/error/code")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    ProviderError {
        kind: provider_error_kind_from_code(code.as_deref()).unwrap_or(ProviderErrorKind::Provider),
        provider,
        status: None,
        code,
        message,
        retry_after: None,
        request_id: None,
        raw: Some(Box::new(raw)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ProviderGenerationOptions;

    fn chat_request(
        options: ProviderGenerationOptions,
        messages: Vec<ChatMessage>,
    ) -> ProviderChatRequest {
        ProviderChatRequest {
            model: "model-a".to_string(),
            messages,
            options,
            provider_options: Default::default(),
        }
    }

    #[test]
    fn tool_call_response_yields_empty_text_with_raw_preserved() {
        let body = serde_json::json!({
            "id": "chatcmpl-1",
            "model": "model-a",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "get_weather", "arguments": "{}" }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });

        let response =
            openai_chat_response_from_body(Some("req-1".to_string()), body, ProviderKind::Proxy)
                .expect("tool-call response should parse");

        assert_eq!(response.result.text, "");
        assert_eq!(
            response.metadata.finish_reason_raw.as_deref(),
            Some("tool_calls")
        );
        assert!(response
            .metadata
            .raw
            .pointer("/choices/0/message/tool_calls")
            .is_some());
    }

    #[test]
    fn rejects_invalid_openai_compatible_request_options() {
        let request = chat_request(
            ProviderGenerationOptions {
                max_tokens: Some(0),
                ..ProviderGenerationOptions::default()
            },
            vec![ChatMessage::new(ChatRole::User, "hello")],
        );
        let err =
            openai_chat_body(&request, ProviderKind::Proxy).expect_err("zero max_tokens fails");
        assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

        let request = chat_request(
            ProviderGenerationOptions {
                temperature: Some(f32::NAN),
                ..ProviderGenerationOptions::default()
            },
            vec![ChatMessage::new(ChatRole::User, "hello")],
        );
        let err =
            openai_chat_body(&request, ProviderKind::Proxy).expect_err("non-finite temp fails");
        assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

        let request = chat_request(ProviderGenerationOptions::default(), Vec::new());
        let err = openai_chat_body(&request, ProviderKind::Proxy).expect_err("empty messages fail");
        assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
    }

    #[test]
    fn sse_parser_handles_partial_events() {
        let mut parser = SseParser::new(ProviderKind::Proxy);

        let first = parser
            .push(br#"data: {"choices":[{"delta":{"content":"he"}"#)
            .expect("partial push");
        assert!(first.is_empty());

        let second = parser
            .push(b"}]}\n\ndata: [DONE]\n\n")
            .expect("complete push");
        assert_eq!(
            second,
            vec![
                r#"{"choices":[{"delta":{"content":"he"}}]}"#.to_string(),
                "[DONE]".to_string()
            ]
        );
    }

    #[test]
    fn sse_parser_flushes_trailing_event() {
        let mut parser = SseParser::new(ProviderKind::Proxy);

        let pushed = parser
            .push(br#"data: {"choices":[{"delta":{"content":"he"}}]}"#)
            .expect("partial push");
        assert!(pushed.is_empty());

        assert_eq!(
            parser.finish(),
            vec![r#"{"choices":[{"delta":{"content":"he"}}]}"#.to_string()]
        );
    }
}
