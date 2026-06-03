//! Tests the `providers::anthropic` module in `cogentlm-providers`.
//!
//! Covers Anthropic provider construction, request mapping, response parsing,
//! usage handling, unsupported embeddings, and SSE stream behavior with
//! deterministic `wiremock` and byte fixtures and no live network calls.

use std::time::Duration;

use bytes::Bytes;
use cogentlm_core::{ChatMessage, ChatRole, FinishReason};
use futures_util::StreamExt;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::{ProviderGenerationOptions, SecretString};

fn config(server: &MockServer) -> AnthropicConfig {
    AnthropicConfig {
        api_key: SecretString::new("token"),
        base_url: Some(server.uri()),
        version: None,
        timeout: None,
    }
}

#[test]
fn rejects_empty_version_and_zero_timeout() {
    let err = match AnthropicProvider::new(AnthropicConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        version: Some(" ".to_string()),
        timeout: None,
    }) {
        Ok(_) => panic!("empty version should be rejected"),
        Err(err) => err,
    };
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = match AnthropicProvider::new(AnthropicConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        version: None,
        timeout: Some(Duration::ZERO),
    }) {
        Ok(_) => panic!("zero timeout should be rejected"),
        Err(err) => err,
    };
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[test]
fn kind_returns_anthropic() {
    let provider = AnthropicProvider::new(AnthropicConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        version: None,
        timeout: None,
    })
    .expect("provider");

    assert_eq!(provider.kind(), ProviderKind::Anthropic);
}

#[tokio::test]
async fn lists_anthropic_models() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("x-api-key", "token"))
        .and(header("anthropic-version", DEFAULT_ANTHROPIC_VERSION))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "data": [{
                "id": "claude-test",
                "type": "model",
                "display_name": "Claude Test"
            }]
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(config(&server)).expect("provider");
    let models = provider.list_models().await.expect("models");

    assert_eq!(models[0].id, "claude-test");
    assert_eq!(models[0].provider, ProviderKind::Anthropic);
    assert_eq!(models[0].display_name.as_deref(), Some("Claude Test"));
    assert_eq!(
        models[0].capabilities.embeddings,
        CapabilitySupport::Unsupported
    );
}

#[tokio::test]
async fn gets_anthropic_model() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models/claude-test"))
        .and(header("x-api-key", "token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "claude-test",
            "type": "model",
            "display_name": "Claude Test"
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(config(&server)).expect("provider");
    let model = provider.get_model("claude-test").await.expect("model");

    assert_eq!(model.id, "claude-test");
    assert_eq!(model.provider, ProviderKind::Anthropic);
}

#[tokio::test]
async fn maps_anthropic_chat_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "token"))
        .and(body_json(json!({
            "model": "claude-test",
            "messages": [{ "role": "user", "content": "hello" }],
            "system": "You are terse.",
            "max_tokens": 16,
            "temperature": 0.5,
            "stop_sequences": ["END"]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("request-id", "req-chat")
                .set_body_json(json!({
                    "id": "msg-test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [{ "type": "text", "text": "hi" }],
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": {
                        "input_tokens": 3,
                        "cache_read_input_tokens": 2,
                        "output_tokens": 4
                    }
                })),
        )
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(config(&server)).expect("provider");
    let response = provider
        .chat(ProviderChatRequest {
            model: "claude-test".to_string(),
            messages: vec![
                ChatMessage::new(ChatRole::System, "You are terse."),
                ChatMessage::new(ChatRole::User, "hello"),
            ],
            options: ProviderGenerationOptions {
                max_tokens: Some(16),
                temperature: Some(0.5),
                stop: vec!["END".to_string()],
                ..ProviderGenerationOptions::default()
            },
            provider_options: Default::default(),
        })
        .await
        .expect("chat");

    assert_eq!(response.result.text, "hi");
    assert_eq!(response.result.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.expect("usage").input_tokens, Some(5));
    assert_eq!(response.metadata.provider, ProviderKind::Anthropic);
    assert_eq!(response.metadata.request_id.as_deref(), Some("req-chat"));
    assert_eq!(response.metadata.response_id.as_deref(), Some("msg-test"));
}

#[tokio::test]
async fn maps_anthropic_generate_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "token"))
        .and(body_json(json!({
            "model": "claude-test",
            "messages": [{ "role": "user", "content": "tell me" }],
            "max_tokens": DEFAULT_ANTHROPIC_MAX_TOKENS
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{ "type": "text", "text": "done" }],
            "stop_reason": "max_tokens",
            "stop_sequence": null,
            "usage": { "input_tokens": 2, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(config(&server)).expect("provider");
    let response = provider
        .generate(ProviderGenerateRequest {
            model: "claude-test".to_string(),
            prompt: "tell me".to_string(),
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("generate");

    assert_eq!(response.result.text, "done");
    assert_eq!(response.result.finish_reason, FinishReason::Length);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(3));
}

#[test]
fn anthropic_chat_body_maps_roles_options_system_join_and_provider_options() {
    let request = ProviderChatRequest {
        model: "claude-test".to_string(),
        messages: vec![
            ChatMessage::new(ChatRole::System, "system one"),
            ChatMessage::new(ChatRole::System, " "),
            ChatMessage::new(ChatRole::System, "system two"),
            ChatMessage::new(ChatRole::User, "hello"),
            ChatMessage::new(ChatRole::Assistant, "hi"),
        ],
        options: ProviderGenerationOptions {
            max_tokens: Some(32),
            temperature: Some(0.25),
            top_p: Some(0.9),
            stop: vec!["END".to_string()],
        },
        provider_options: [("metadata".to_string(), json!({ "source": "test" }))]
            .into_iter()
            .collect(),
    };

    let body = anthropic_chat_body(&request, true).expect("chat body");

    assert_eq!(body["model"], "claude-test");
    assert_eq!(body["system"], "system one\n\nsystem two");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][1]["role"], "assistant");
    assert_eq!(body["max_tokens"], 32);
    assert_eq!(body["temperature"], json!(0.25));
    assert_eq!(body["top_p"], json!(0.9_f32));
    assert_eq!(body["stop_sequences"], json!(["END"]));
    assert_eq!(body["stream"], true);
    assert_eq!(body["metadata"]["source"], "test");
}

#[test]
fn rejects_invalid_anthropic_chat_body_options() {
    let base = ProviderChatRequest {
        model: "claude-test".to_string(),
        messages: vec![ChatMessage::new(ChatRole::User, "hello")],
        options: ProviderGenerationOptions::default(),
        provider_options: Default::default(),
    };

    let mut request = base.clone();
    request.model = " ".to_string();
    let err = anthropic_chat_body(&request, false).expect_err("blank model");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let mut request = base.clone();
    request.options.max_tokens = Some(0);
    let err = anthropic_chat_body(&request, false).expect_err("zero max tokens");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let mut request = base.clone();
    request.options.temperature = Some(f32::NAN);
    let err = anthropic_chat_body(&request, false).expect_err("nan temperature");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let mut request = base.clone();
    request.options.top_p = Some(f32::INFINITY);
    let err = anthropic_chat_body(&request, false).expect_err("infinite top_p");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let mut request = base;
    request
        .provider_options
        .insert("stream".to_string(), json!(true));
    let err = anthropic_chat_body(&request, false).expect_err("typed collision");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[tokio::test]
async fn maps_anthropic_non_text_response_without_claiming_tool_support() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/messages"))
        .and(header("x-api-key", "token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "model": "claude-test",
            "content": [{
                "type": "tool_use",
                "id": "toolu_test",
                "name": "get_weather",
                "input": { "location": "Toronto" }
            }],
            "stop_reason": "tool_use",
            "stop_sequence": null,
            "usage": { "input_tokens": 2, "output_tokens": 1 }
        })))
        .mount(&server)
        .await;

    let provider = AnthropicProvider::new(config(&server)).expect("provider");
    let response = provider
        .chat(ProviderChatRequest {
            model: "claude-test".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: [(
                "tools".to_string(),
                json!([{
                    "name": "get_weather",
                    "input_schema": { "type": "object" }
                }]),
            )]
            .into_iter()
            .collect(),
        })
        .await
        .expect("chat");

    assert_eq!(response.result.text, "");
    assert_eq!(response.result.finish_reason, FinishReason::Stop);
    assert_eq!(
        response.metadata.finish_reason_raw.as_deref(),
        Some("tool_use")
    );
    assert_eq!(response.metadata.raw["content"][0]["type"], "tool_use");
}

#[test]
fn rejects_malformed_anthropic_model_and_text_responses() {
    let err = anthropic_models_from_body(&json!({ "data": null }))
        .expect_err("missing data array should fail");
    assert_eq!(err.kind, ProviderErrorKind::Provider);

    let err = anthropic_model_from_value(&json!({ "type": "model" }))
        .expect_err("missing model id should fail");
    assert_eq!(err.kind, ProviderErrorKind::Provider);

    let err = anthropic_text_response(
        None,
        json!({
            "error": {
                "message": "busy",
                "type": "overloaded_error"
            }
        }),
    )
    .expect_err("body error should fail");
    assert_eq!(err.kind, ProviderErrorKind::Overloaded);

    for body in [
        json!({ "model": "claude-test" }),
        json!({ "content": [{ "type": "text", "text": "hi" }] }),
        json!({ "model": "claude-test", "content": [{ "type": "text" }] }),
        json!({
            "model": "claude-test",
            "content": [{ "type": "text", "text": "hi" }],
            "usage": { "input_tokens": "bad" }
        }),
        json!({
            "model": "claude-test",
            "content": [{ "type": "text", "text": "hi" }],
            "usage": {
                "input_tokens": u32::MAX,
                "cache_read_input_tokens": 1
            }
        }),
    ] {
        let err =
            anthropic_text_response(None, body).expect_err("malformed text response should fail");
        assert_eq!(err.kind, ProviderErrorKind::Provider);
    }
}

#[test]
fn anthropic_usage_includes_cache_creation_tokens() {
    let response = anthropic_text_response(
        None,
        json!({
            "id": "msg-test",
            "model": "claude-test",
            "content": [{ "type": "text", "text": "hi" }],
            "usage": {
                "input_tokens": 1,
                "cache_creation_input_tokens": 2,
                "output_tokens": 3
            }
        }),
    )
    .expect("response");

    let usage = response.usage.expect("usage");
    assert_eq!(usage.input_tokens, Some(3));
    assert_eq!(usage.output_tokens, Some(3));
    assert_eq!(usage.total_tokens, Some(6));
}

#[tokio::test]
async fn rejects_provider_options_colliding_with_typed_fields() {
    let provider = AnthropicProvider::new(AnthropicConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        version: None,
        timeout: None,
    })
    .expect("provider");

    let err = provider
        .chat(ProviderChatRequest {
            model: "claude-test".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: [("model".to_string(), json!("other"))]
                .into_iter()
                .collect(),
        })
        .await
        .expect_err("provider_options model should be rejected");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[tokio::test]
async fn rejects_system_only_chat() {
    let provider = AnthropicProvider::new(AnthropicConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        version: None,
        timeout: None,
    })
    .expect("provider");

    let err = provider
        .chat(ProviderChatRequest {
            model: "claude-test".to_string(),
            messages: vec![ChatMessage::new(ChatRole::System, "You are terse.")],
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect_err("system-only chat should fail");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[tokio::test]
async fn rejects_embeddings() {
    let provider = AnthropicProvider::new(AnthropicConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        version: None,
        timeout: None,
    })
    .expect("provider");

    let err = provider
        .embed(ProviderEmbedRequest {
            model: "claude-test".to_string(),
            input: "hello".to_string(),
            provider_options: Default::default(),
        })
        .await
        .expect_err("embeddings should fail");

    assert_eq!(err.kind, ProviderErrorKind::UnsupportedFeature);
}

#[tokio::test]
async fn streams_anthropic_messages() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "token"))
            .and(body_json(json!({
                "model": "claude-test",
                "messages": [{ "role": "user", "content": "hello" }],
                "max_tokens": DEFAULT_ANTHROPIC_MAX_TOKENS,
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("request-id", "req-stream")
                    .set_body_string(concat!(
                        "event: message_start\n",
                        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg-test\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-test\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n",
                        "event: content_block_delta\n",
                        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hel\"}}\n\n",
                        "event: ping\n",
                        "data: {\"type\":\"ping\"}\n\n",
                        "event: content_block_delta\n",
                        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
                        "event: message_delta\n",
                        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":2}}\n\n",
                        "event: message_delta\n",
                        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null}}\n\n",
                        "event: message_stop\n",
                        "data: {\"type\":\"message_stop\"}\n\n"
                    )),
            )
            .mount(&server)
            .await;

    let provider = AnthropicProvider::new(config(&server)).expect("provider");
    let mut stream = provider
        .stream_chat(ProviderChatRequest {
            model: "claude-test".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("stream");

    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event.expect("stream event"));
    }

    assert_eq!(
        events,
        vec![
            ProviderStreamEvent::Usage {
                usage: TokenUsage {
                    input_tokens: Some(3),
                    output_tokens: Some(1),
                    total_tokens: Some(4)
                }
            },
            ProviderStreamEvent::TokenBatch(
                TokenBatchBuilder::new(Some("req-stream".to_string())).push_text("hel")
            ),
            ProviderStreamEvent::TokenBatch({
                let mut builder = TokenBatchBuilder::new(Some("req-stream".to_string()));
                builder.push_text("hel");
                builder.push_text("lo")
            }),
            ProviderStreamEvent::Usage {
                usage: TokenUsage {
                    input_tokens: Some(3),
                    output_tokens: Some(2),
                    total_tokens: Some(5)
                }
            },
            ProviderStreamEvent::Finished {
                finish_reason: FinishReason::Stop
            }
        ]
    );
}

#[tokio::test]
async fn anthropic_stream_events_handle_stop_without_delta_and_ignored_events() {
    let events = anthropic_stream_events(
        Some("req-stop".to_string()),
        byte_stream(vec![Ok(Bytes::from_static(
            concat!(
                "event: content_block_start\n",
                "data: {\"type\":\"content_block_start\"}\n\n",
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0}\n\n",
                "event: unknown\n",
                "data: {\"type\":\"unknown_event\"}\n\n",
                "event: message_stop\n",
                "data: {\"type\":\"message_stop\"}\n\n"
            )
            .as_bytes(),
        ))]),
    )
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<ProviderResult<Vec<_>>>()
    .expect("events");

    assert_eq!(
        events,
        vec![ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }]
    );
}

#[tokio::test]
async fn anthropic_stream_events_surface_invalid_json_missing_text_and_byte_errors() {
    let mut stream = anthropic_stream_events(
        Some("req-invalid".to_string()),
        byte_stream(vec![Ok(Bytes::from_static(b"data: {bad json}\n\n"))]),
    );
    let err = stream
        .next()
        .await
        .expect("event")
        .expect_err("invalid json should fail");
    assert_eq!(err.kind, ProviderErrorKind::Provider);
    assert_eq!(err.request_id.as_deref(), Some("req-invalid"));

    let mut stream = anthropic_stream_events(
        Some("req-missing-text".to_string()),
        byte_stream(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\"}}\n\n",
        ))]),
    );
    let err = stream
        .next()
        .await
        .expect("event")
        .expect_err("missing text should fail");
    assert_eq!(err.kind, ProviderErrorKind::Provider);
    assert_eq!(err.request_id.as_deref(), Some("req-missing-text"));

    let mut stream = anthropic_stream_events(
        Some("req-byte-error".to_string()),
        byte_stream(vec![Err(ProviderError::new(
            ProviderErrorKind::Transport,
            ProviderKind::Anthropic,
            "broken stream",
        ))]),
    );
    let err = stream
        .next()
        .await
        .expect("event")
        .expect_err("byte error should fail");
    assert_eq!(err.kind, ProviderErrorKind::Transport);
}

#[tokio::test]
async fn anthropic_stream_events_flush_trailing_payloads_and_errors_at_stream_end() {
    let events = anthropic_stream_events(
        Some("req-trailing".to_string()),
        byte_stream(vec![Ok(Bytes::from_static(
            b"data: {\"type\":\"message_stop\"}",
        ))]),
    )
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<ProviderResult<Vec<_>>>()
    .expect("events");

    assert_eq!(
        events,
        vec![ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }]
    );

    let mut stream = anthropic_stream_events(
        Some("req-trailing-error".to_string()),
        byte_stream(vec![Ok(Bytes::from_static(b"data: {bad json}"))]),
    );
    let err = stream
        .next()
        .await
        .expect("event")
        .expect_err("trailing invalid json should fail");

    assert_eq!(err.kind, ProviderErrorKind::Provider);
    assert_eq!(err.request_id.as_deref(), Some("req-trailing-error"));
}

#[tokio::test]
async fn maps_anthropic_stream_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("request-id", "req-stream-error")
                    .set_body_string(concat!(
                        "event: error\n",
                        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n"
                    )),
            )
            .mount(&server)
            .await;

    let provider = AnthropicProvider::new(config(&server)).expect("provider");
    let mut stream = provider
        .stream_chat(ProviderChatRequest {
            model: "claude-test".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("stream");
    let err = stream
        .next()
        .await
        .expect("first event")
        .expect_err("error event should fail");

    assert_eq!(err.kind, ProviderErrorKind::Overloaded);
    assert_eq!(err.code.as_deref(), Some("overloaded_error"));
    assert_eq!(err.request_id.as_deref(), Some("req-stream-error"));
}

fn byte_stream(items: Vec<ProviderResult<Bytes>>) -> HttpByteStream {
    Box::pin(futures_util::stream::iter(items))
}
