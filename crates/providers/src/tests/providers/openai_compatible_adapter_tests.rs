//! Tests the `providers::proxy` module in `cogentlm-providers`.
//!
//! Covers OpenAI-compatible proxy provider construction, request mapping,
//! response parsing, unsupported paths, HTTP error handling, and stream behavior
//! with deterministic `wiremock` fixtures and no live network calls.

use std::time::Duration;

use bytes::Bytes;
use cogentlm_core::{ChatMessage, ChatRole, FinishReason, TokenBatch};
use futures_util::StreamExt;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::{
    OpenAiCompatibleProtocol, ProviderAuth, ProviderError, ProviderErrorKind,
    ProviderGenerationOptions, ProviderRequestContext, SecretString, TokenUsage,
};

fn config(server: &MockServer) -> OpenAiCompatibleAdapterConfig {
    OpenAiCompatibleAdapterConfig {
        base_url: server.uri(),
        auth: ProviderAuth::Bearer(SecretString::new("token")),
        protocol: OpenAiCompatibleProtocol::OpenAiCompatible,
        static_headers: Vec::new(),
        correlation_header: None,
        timeout: None,
    }
}

#[test]
fn rejects_zero_timeout() {
    let mut config = OpenAiCompatibleAdapterConfig {
        base_url: "http://localhost".to_string(),
        auth: ProviderAuth::Bearer(SecretString::new("token")),
        protocol: OpenAiCompatibleProtocol::OpenAiCompatible,
        static_headers: Vec::new(),
        correlation_header: None,
        timeout: Some(Duration::ZERO),
    };

    let err = match OpenAiCompatibleAdapter::new(config.clone()) {
        Ok(_) => panic!("zero timeout should be rejected"),
        Err(err) => err,
    };
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    config.timeout = Some(Duration::from_millis(1));
    OpenAiCompatibleAdapter::new(config).expect("positive timeout");
}

#[tokio::test]
async fn configurable_correlation_header_is_sent() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/completions"))
        .and(header("x-trace-id", "gateway-request-2"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "cmpl-test",
            "model": "gpt-test",
            "choices": [{ "text": "done", "finish_reason": "stop" }]
        })))
        .mount(&server)
        .await;
    let mut adapter_config = config(&server);
    adapter_config.correlation_header = Some("x-trace-id".to_string());
    let provider = OpenAiCompatibleAdapter::new(adapter_config).expect("provider");

    provider
        .generate_with_context(
            ProviderRequestContext {
                request_id: Some("gateway-request-2".to_string()),
            },
            ProviderGenerateRequest {
                model: "gpt-test".to_string(),
                prompt: "hello".to_string(),
                options: ProviderGenerationOptions::default(),
                provider_options: Default::default(),
            },
        )
        .await
        .expect("completion");
}

#[tokio::test]
async fn lists_openai_compatible_models() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", "Bearer token"))
        .and(header("x-cogent-test", "yes"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [
                { "id": "model-a", "object": "model" },
                { "id": "model-b", "object": "model" }
            ]
        })))
        .mount(&server)
        .await;

    let mut config = config(&server);
    config
        .static_headers
        .push(("x-cogent-test".to_string(), "yes".to_string()));
    let provider = OpenAiCompatibleAdapter::new(config).expect("OpenAI-compatible adapter");
    let models = provider.list_models().await.expect("models");

    assert_eq!(models.len(), 2);
    assert_eq!(models[0].id, "model-a");
    assert_eq!(models[0].raw["object"], "model");
}

#[tokio::test]
async fn gets_openai_compatible_model() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models/model-a"))
        .and(header("authorization", "Bearer token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "model-a",
            "object": "model"
        })))
        .mount(&server)
        .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let model = provider.get_model("model-a").await.expect("model");

    assert_eq!(model.id, "model-a");
    assert_eq!(model.provider, ProviderKind::OpenAiCompatible);
    assert_eq!(model.raw["object"], "model");
}

#[tokio::test]
async fn maps_openai_compatible_embeddings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "model-a",
            "input": "hello",
            "encoding_format": "float"
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-embed")
                .set_body_json(json!({
                    "object": "list",
                    "model": "model-a",
                    "data": [{
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.25, -0.5]
                    }],
                    "usage": {
                        "prompt_tokens": 2,
                        "total_tokens": 2
                    }
                })),
        )
        .mount(&server)
        .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let response = provider
        .embed(ProviderEmbedRequest {
            model: "model-a".to_string(),
            input: "hello".to_string(),
            provider_options: Default::default(),
        })
        .await
        .expect("embeddings");

    assert_eq!(response.result.values, vec![0.25, -0.5]);
    assert_eq!(response.usage.expect("usage").input_tokens, Some(2));
    assert_eq!(response.metadata.provider, ProviderKind::OpenAiCompatible);
    assert_eq!(response.metadata.request_id.as_deref(), Some("req-embed"));
}

#[tokio::test]
async fn maps_openai_compatible_chat_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "model-a",
            "messages": [
                { "role": "user", "content": "hello" }
            ],
            "max_tokens": 16
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("request-id", "req-1")
                .set_body_json(json!({
                    "id": "chatcmpl-1",
                    "model": "model-a",
                    "choices": [
                        {
                            "message": { "role": "assistant", "content": "hi" },
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5
                    }
                })),
        )
        .mount(&server)
        .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let response = provider
        .chat(ProviderChatRequest {
            model: "model-a".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions {
                max_tokens: Some(16),
                ..ProviderGenerationOptions::default()
            },
            provider_options: Default::default(),
        })
        .await
        .expect("chat");

    assert_eq!(response.result.text, "hi");
    assert_eq!(response.result.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(5));
    assert_eq!(response.metadata.request_id.as_deref(), Some("req-1"));
    assert_eq!(response.metadata.response_id.as_deref(), Some("chatcmpl-1"));
    assert_eq!(response.metadata.finish_reason_raw.as_deref(), Some("stop"));
}

#[tokio::test]
async fn maps_openai_compatible_completion_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/completions"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "model-a",
            "prompt": "hello",
            "max_tokens": 16,
            "stop": ["END"]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("request-id", "req-completion")
                .set_body_json(json!({
                    "id": "cmpl-1",
                    "model": "model-a",
                    "choices": [
                        {
                            "text": "hi",
                            "finish_reason": "stop"
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5
                    }
                })),
        )
        .mount(&server)
        .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let response = provider
        .generate(ProviderGenerateRequest {
            model: "model-a".to_string(),
            prompt: "hello".to_string(),
            options: ProviderGenerationOptions {
                max_tokens: Some(16),
                stop: vec!["END".to_string()],
                ..ProviderGenerationOptions::default()
            },
            provider_options: Default::default(),
        })
        .await
        .expect("completion");

    assert_eq!(response.result.text, "hi");
    assert_eq!(response.result.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(5));
    assert_eq!(
        response.metadata.request_id.as_deref(),
        Some("req-completion")
    );
    assert_eq!(response.metadata.response_id.as_deref(), Some("cmpl-1"));
}

#[tokio::test]
async fn streams_openai_compatible_completion_chunks() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .and(path("/completions"))
            .and(header("authorization", "Bearer token"))
            .and(body_json(json!({
                "model": "model-a",
                "prompt": "hello",
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-request-id", "req-stream-completion")
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(concat!(
                        "data: {\"id\":\"cmpl-1\",\"model\":\"model-a\",\"choices\":[{\"text\":\"he\",\"finish_reason\":null}]}\n\n",
                        "data: {\"id\":\"cmpl-1\",\"model\":\"model-a\",\"choices\":[{\"text\":\"llo\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n",
                        "data: [DONE]\n\n"
                    )),
            )
            .mount(&server)
            .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let events = provider
        .stream_generate(ProviderGenerateRequest {
            model: "model-a".to_string(),
            prompt: "hello".to_string(),
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("completion stream")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<ProviderResult<Vec<_>>>()
        .expect("events");

    assert!(matches!(
        &events[0],
        ProviderStreamEvent::TokenBatch(TokenBatch { text, sequence_start, .. })
            if text == "he" && *sequence_start == 0
    ));
    assert!(matches!(
        &events[1],
        ProviderStreamEvent::Usage {
            usage: TokenUsage {
                total_tokens: Some(3),
                ..
            }
        }
    ));
    assert!(matches!(
        &events[2],
        ProviderStreamEvent::TokenBatch(TokenBatch { text, sequence_start, .. })
            if text == "llo" && *sequence_start == 1
    ));
    assert!(matches!(
        events[3],
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    ));
}

#[tokio::test]
async fn rejects_provider_options_colliding_with_typed_fields() {
    let server = MockServer::start().await;
    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");

    let err = provider
        .chat(ProviderChatRequest {
            model: "model-a".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: [("model".to_string(), json!("other"))]
                .into_iter()
                .collect(),
        })
        .await
        .expect_err("collision should fail");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = provider
        .generate(ProviderGenerateRequest {
            model: "model-a".to_string(),
            prompt: "hello".to_string(),
            options: ProviderGenerationOptions::default(),
            provider_options: [("prompt".to_string(), json!("other"))]
                .into_iter()
                .collect(),
        })
        .await
        .expect_err("completion typed field collision should fail");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = provider
        .embed(ProviderEmbedRequest {
            model: "model-a".to_string(),
            input: "hello".to_string(),
            provider_options: [("input".to_string(), json!("other"))]
                .into_iter()
                .collect(),
        })
        .await
        .expect_err("embedding typed field collision should fail");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let err = provider
        .chat(ProviderChatRequest {
            model: "model-a".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: [("stop".to_string(), json!(["END"]))].into_iter().collect(),
        })
        .await
        .expect_err("optional typed field collision should fail");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[tokio::test]
async fn maps_provider_http_errors() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(
            ResponseTemplate::new(429)
                .insert_header("retry-after-ms", "1500")
                .set_body_json(json!({
                    "error": {
                        "message": "rate limited",
                        "code": "rate_limit"
                    }
                })),
        )
        .mount(&server)
        .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let err = provider
        .list_models()
        .await
        .expect_err("rate limit should fail");

    assert_eq!(err.kind, ProviderErrorKind::RateLimited);
    assert_eq!(err.status, Some(429));
    assert_eq!(err.code.as_deref(), Some("rate_limit"));
    assert_eq!(err.retry_after, Some(Duration::from_millis(1500)));
}

#[tokio::test]
async fn maps_openai_compatible_chat_error_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "model-a",
            "messages": [
                { "role": "user", "content": "hello" }
            ]
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "error": {
                "message": "slow down",
                "code": "rate_limit"
            }
        })))
        .mount(&server)
        .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let err = provider
        .chat(ProviderChatRequest {
            model: "model-a".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect_err("provider error body should fail");

    assert_eq!(err.kind, ProviderErrorKind::RateLimited);
    assert_eq!(err.code.as_deref(), Some("rate_limit"));
    assert_eq!(err.message, "slow down");
    assert!(err.raw.is_some());
}

#[tokio::test]
async fn classifies_non_json_error_body_by_status() {
    // Gateways/CDNs often return HTML or plain text on 5xx; the error must
    // still be classified by HTTP status, not collapsed to a transport error.
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .respond_with(
            ResponseTemplate::new(503)
                .insert_header("content-type", "text/html")
                .set_body_string("<html><body>503 Service Unavailable</body></html>"),
        )
        .mount(&server)
        .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let err = provider.list_models().await.expect_err("503 should fail");

    assert_eq!(err.kind, ProviderErrorKind::Overloaded);
    assert_eq!(err.status, Some(503));
    assert!(err.raw.is_some());
}

#[tokio::test]
async fn stream_maps_provider_error_event() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer token"))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "data: {\"error\":{\"message\":\"slow down\",\"code\":\"rate_limit\"}}\n\n",
                    "data: [DONE]\n\n"
                )),
        )
        .mount(&server)
        .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let mut stream = provider
        .stream_chat(ProviderChatRequest {
            model: "model-a".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("stream");
    let err = stream
        .next()
        .await
        .expect("error event")
        .expect_err("provider error event should fail");

    assert_eq!(err.kind, ProviderErrorKind::RateLimited);
    assert_eq!(err.code.as_deref(), Some("rate_limit"));
    assert!(err.raw.is_some());
}

#[tokio::test]
async fn streams_openai_compatible_chat_chunks() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer token"))
            .and(body_json(json!({
                "model": "model-a",
                "messages": [
                    { "role": "user", "content": "hello" }
                ],
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-request-id", "req-stream")
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(concat!(
                        "data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"delta\":{\"content\":\"he\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"id\":\"chatcmpl-1\",\"model\":\"model-a\",\"choices\":[{\"delta\":{\"content\":\"llo\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":2,\"total_tokens\":3}}\n\n",
                        "data: [DONE]\n\n"
                    )),
            )
            .mount(&server)
            .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let events = provider
        .stream_chat(ProviderChatRequest {
            model: "model-a".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("stream")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<ProviderResult<Vec<_>>>()
        .expect("events");

    assert!(matches!(
        &events[0],
        ProviderStreamEvent::TokenBatch(TokenBatch { text, sequence_start, .. })
            if text == "he" && *sequence_start == 0
    ));
    assert!(matches!(
        &events[1],
        ProviderStreamEvent::Usage {
            usage: TokenUsage {
                total_tokens: Some(3),
                ..
            }
        }
    ));
    assert!(matches!(
        &events[2],
        ProviderStreamEvent::TokenBatch(TokenBatch { text, sequence_start, .. })
            if text == "llo" && *sequence_start == 1
    ));
    assert!(matches!(
        events[3],
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    ));
}

#[tokio::test]
async fn stream_ignores_null_error_field() {
    let chunks = futures_util::stream::iter([Ok::<_, ProviderError>(Bytes::from_static(
        concat!(
            "data: {\"error\":null,\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        )
        .as_bytes(),
    ))]);
    let events = openai_stream_events(
        Some("req-null-error".to_string()),
        Box::pin(chunks),
        ProviderKind::OpenAiCompatible,
    )
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<ProviderResult<Vec<_>>>()
    .expect("events");

    assert!(matches!(
        &events[0],
        ProviderStreamEvent::TokenBatch(TokenBatch { text, .. }) if text == "ok"
    ));
    assert!(matches!(
        events[1],
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    ));
}

#[tokio::test]
async fn openai_compatible_stream_rejects_eof_before_finish() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer token"))
            .and(body_json(json!({
                "model": "model-a",
                "messages": [
                    { "role": "user", "content": "hello" }
                ],
                "stream": true
            })))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("x-request-id", "req-truncated")
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n",
                    ),
            )
            .mount(&server)
            .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let mut stream = provider
        .stream_chat(ProviderChatRequest {
            model: "model-a".to_string(),
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
        .expect("missing finish error")
        .expect_err("truncated provider stream must fail");

    assert!(matches!(
        first,
        ProviderStreamEvent::TokenBatch(TokenBatch { text, .. }) if text == "partial"
    ));
    assert_eq!(error.kind, ProviderErrorKind::Provider);
    assert_eq!(
        error.message,
        "OpenAI-compatible stream ended before finish_reason"
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn stream_rejects_non_object_payload() {
    let chunks = futures_util::stream::iter([Ok::<_, ProviderError>(Bytes::from_static(
        concat!("data: []\n\n", "data: [DONE]\n\n").as_bytes(),
    ))]);
    let mut stream = openai_stream_events(
        Some("req-array-payload".to_string()),
        Box::pin(chunks),
        ProviderKind::OpenAiCompatible,
    );

    let error = stream
        .next()
        .await
        .expect("array payload error")
        .expect_err("array payload should fail");

    assert_eq!(error.kind, ProviderErrorKind::Provider);
    assert_eq!(
        error.message,
        "OpenAI-compatible stream payload must be a JSON object"
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn stream_rejects_payloads_after_finish_reason() {
    let chunks = futures_util::stream::iter([
        Ok::<_, ProviderError>(Bytes::from_static(
            concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
                "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n",
            )
            .as_bytes(),
        )),
        Ok(Bytes::from_static(
            b"data: {\"choices\":[{\"delta\":{\"content\":\"late\"},\"finish_reason\":null}]}\n\n",
        )),
    ]);
    let mut stream = openai_stream_events(
        Some("req-after-finish".to_string()),
        Box::pin(chunks),
        ProviderKind::OpenAiCompatible,
    );

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
        ProviderStreamEvent::TokenBatch(TokenBatch { text, .. }) if text == "hi"
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
        "OpenAI-compatible stream event received after finish_reason"
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn stream_rejects_choice_text_after_finish_reason_in_same_payload() {
    let chunks = futures_util::stream::iter([Ok::<_, ProviderError>(Bytes::from_static(
        concat!(
            "data: {\"choices\":[",
            "{\"delta\":{\"content\":\"hi\"},\"finish_reason\":\"stop\"},",
            "{\"delta\":{\"content\":\"late\"},\"finish_reason\":null}",
            "]}\n\n",
        )
        .as_bytes(),
    ))]);
    let mut stream = openai_stream_events(
        Some("req-same-payload-after-finish".to_string()),
        Box::pin(chunks),
        ProviderKind::OpenAiCompatible,
    );

    let error = stream
        .next()
        .await
        .expect("same-payload error")
        .expect_err("late choice text should fail");

    assert_eq!(error.kind, ProviderErrorKind::Provider);
    assert_eq!(
        error.message,
        "OpenAI-compatible stream event received after finish_reason"
    );
    assert!(stream.next().await.is_none());
}

#[tokio::test]
async fn stream_skips_null_usage_chunks() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
            .and(path("/chat/completions"))
            .and(header("authorization", "Bearer token"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(concat!(
                        "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}],\"usage\":null}\n\n",
                        "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":null}\n\n",
                        "data: [DONE]\n\n"
                    )),
            )
            .mount(&server)
            .await;

    let provider =
        OpenAiCompatibleAdapter::new(config(&server)).expect("OpenAI-compatible adapter");
    let events = provider
        .stream_chat(ProviderChatRequest {
            model: "model-a".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("stream")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<ProviderResult<Vec<_>>>()
        .expect("events");

    assert!(!events
        .iter()
        .any(|event| matches!(event, ProviderStreamEvent::Usage { .. })));
    assert!(matches!(
        &events[0],
        ProviderStreamEvent::TokenBatch(TokenBatch { text, .. }) if text == "hi"
    ));
    assert!(matches!(
        events[1],
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    ));
}
