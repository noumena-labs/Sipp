//! Unit tests for the parent module.

use std::time::Duration;

use bytes::Bytes;
use cogentlm_core::{ChatMessage, ChatRole, FinishReason};
use futures_util::StreamExt;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::{ProviderError, ProviderGenerationOptions, SecretString};

fn config(server: &MockServer) -> AnthropicAdapterConfig {
    AnthropicAdapterConfig {
        api_key: SecretString::new("token"),
        base_url: Some(server.uri()),
        version: None,
        timeout: None,
    }
}

#[test]
fn rejects_empty_version_and_zero_timeout() {
    let err = match AnthropicAdapter::new(AnthropicAdapterConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        version: Some(" ".to_string()),
        timeout: None,
    }) {
        Ok(_) => panic!("empty version should be rejected"),
        Err(err) => err,
    };
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = match AnthropicAdapter::new(AnthropicAdapterConfig {
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

    let provider = AnthropicAdapter::new(config(&server)).expect("provider");
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

    let provider = AnthropicAdapter::new(config(&server)).expect("provider");
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

    let provider = AnthropicAdapter::new(config(&server)).expect("provider");
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

    let provider = AnthropicAdapter::new(config(&server)).expect("provider");
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
fn rejects_non_object_anthropic_usage() {
    let body = json!({
        "id": "msg-test",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": [{ "type": "text", "text": "done" }],
        "stop_reason": "end_turn",
        "usage": []
    });

    let err = anthropic_text_response(None, body).expect_err("array usage should fail");

    assert_eq!(err.kind, ProviderErrorKind::Provider);
    assert_eq!(err.message, "usage must be a JSON object");
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

    let provider = AnthropicAdapter::new(config(&server)).expect("provider");
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

#[tokio::test]
async fn rejects_provider_options_colliding_with_typed_fields() {
    let provider = AnthropicAdapter::new(AnthropicAdapterConfig {
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
    let provider = AnthropicAdapter::new(AnthropicAdapterConfig {
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
    let provider = AnthropicAdapter::new(AnthropicAdapterConfig {
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

    let provider = AnthropicAdapter::new(config(&server)).expect("provider");
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
async fn anthropic_stream_rejects_eof_before_message_stop() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .and(path("/messages"))
            .and(header("x-api-key", "token"))
            .and(header("anthropic-version", DEFAULT_ANTHROPIC_VERSION))
            .and(body_json(json!({
                "model": "claude-test",
                "messages": [
                    { "role": "user", "content": "hello" }
                ],
                "max_tokens": DEFAULT_ANTHROPIC_MAX_TOKENS,
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .insert_header("request-id", "req-truncated")
                    .set_body_string(concat!(
                        "event: content_block_delta\n",
                        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"partial\"}}\n\n"
                    )),
            )
            .mount(&server)
            .await;

    let provider = AnthropicAdapter::new(config(&server)).expect("provider");
    let mut stream = provider
        .stream_chat(ProviderChatRequest {
            model: "claude-test".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("stream");
    let first = stream
        .next()
        .await
        .expect("token event")
        .expect("token event");
    let error = stream
        .next()
        .await
        .expect("missing stop error")
        .expect_err("truncated provider stream must fail");

    assert!(matches!(
        first,
        ProviderStreamEvent::TokenBatch(batch) if batch.text == "partial"
    ));
    assert_eq!(error.kind, ProviderErrorKind::Provider);
    assert_eq!(error.message, "Anthropic stream ended before message_stop");
    assert_eq!(error.request_id.as_deref(), Some("req-truncated"));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn anthropic_stream_rejects_non_object_payload() {
    let chunks = futures_util::stream::iter([Ok::<_, ProviderError>(Bytes::from_static(
        concat!(
            "event: ping\n",
            "data: []\n\n",
            "event: message_stop\n",
            "data: {\"type\":\"message_stop\"}\n\n"
        )
        .as_bytes(),
    ))]);
    let mut stream =
        anthropic_stream_events(Some("req-array-payload".to_string()), Box::pin(chunks));

    let error = stream
        .next()
        .await
        .expect("array payload error")
        .expect_err("array payload should fail");

    assert_eq!(error.kind, ProviderErrorKind::Provider);
    assert_eq!(
        error.message,
        "Anthropic stream payload must be a JSON object"
    );
    assert_eq!(error.request_id.as_deref(), Some("req-array-payload"));
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn stream_rejects_payloads_after_message_stop() {
    let chunks = futures_util::stream::iter([
        Ok::<_, ProviderError>(Bytes::from_static(
            concat!(
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n",
                "event: message_stop\n",
                "data: {\"type\":\"message_stop\"}\n\n",
            )
            .as_bytes(),
        )),
        Ok(Bytes::from_static(
            concat!(
                "event: content_block_delta\n",
                "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"late\"}}\n\n",
            )
            .as_bytes(),
        )),
    ]);
    let mut stream = anthropic_stream_events(Some("req-after-stop".to_string()), Box::pin(chunks));

    let first = stream
        .next()
        .await
        .expect("token event")
        .expect("token event");
    let finished = stream
        .next()
        .await
        .expect("finish event")
        .expect("finish event");
    let error = stream
        .next()
        .await
        .expect("late event error")
        .expect_err("late payload should fail");

    assert!(matches!(
        first,
        ProviderStreamEvent::TokenBatch(batch) if batch.text == "hi"
    ));
    assert!(matches!(
        finished,
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    ));
    assert_eq!(error.kind, ProviderErrorKind::Provider);
    assert_eq!(
        error.message,
        "Anthropic stream event received after message_stop"
    );
    assert_eq!(error.request_id.as_deref(), Some("req-after-stop"));
    assert!(stream.next().await.is_none());
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

    let provider = AnthropicAdapter::new(config(&server)).expect("provider");
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
