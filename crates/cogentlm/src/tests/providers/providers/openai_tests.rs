//! Tests the `providers::openai` module in `cogentlm::providers`.
//!
//! Covers OpenAI provider construction, request mapping, response parsing,
//! error paths, and stream routing with deterministic `wiremock` fixtures and
//! no live network calls.

use std::time::Duration;

use crate::core::{ChatMessage, ChatRole, FinishReason};
use futures_util::StreamExt;
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::providers::{
    ProviderErrorKind, ProviderGenerationOptions, ProviderRequestContext, SecretString,
};

fn config(server: &MockServer) -> OpenAiAdapterConfig {
    OpenAiAdapterConfig {
        api_key: SecretString::new("token"),
        base_url: Some(server.uri()),
        timeout: None,
    }
}

#[test]
fn rejects_zero_timeout() {
    let mut config = OpenAiAdapterConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        timeout: Some(Duration::ZERO),
    };

    let err = match OpenAiAdapter::new(config.clone()) {
        Ok(_) => panic!("zero timeout should be rejected"),
        Err(err) => err,
    };
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    config.timeout = Some(Duration::from_millis(1));
    OpenAiAdapter::new(config).expect("positive timeout");
}

#[test]
fn kind_returns_openai() {
    let provider = OpenAiAdapter::new(OpenAiAdapterConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        timeout: None,
    })
    .expect("provider");

    assert_eq!(provider.kind(), ProviderKind::OpenAi);
}

#[tokio::test]
async fn lists_openai_models() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models"))
        .and(header("authorization", "Bearer token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "data": [{ "id": "gpt-test", "object": "model" }]
        })))
        .mount(&server)
        .await;

    let provider = OpenAiAdapter::new(config(&server)).expect("provider");
    let models = provider.list_models().await.expect("models");

    assert_eq!(models[0].id, "gpt-test");
    assert_eq!(models[0].provider, ProviderKind::OpenAi);
}

#[tokio::test]
async fn gets_openai_model() {
    let server = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/models/gpt-test"))
        .and(header("authorization", "Bearer token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "gpt-test",
            "object": "model"
        })))
        .mount(&server)
        .await;

    let provider = OpenAiAdapter::new(config(&server)).expect("provider");
    let model = provider.get_model("gpt-test").await.expect("model");

    assert_eq!(model.id, "gpt-test");
    assert_eq!(model.provider, ProviderKind::OpenAi);
}

#[tokio::test]
async fn maps_openai_embeddings() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/embeddings"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "gpt-test",
            "input": "hello",
            "encoding_format": "float"
        })))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "object": "list",
            "model": "gpt-test",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.125, -0.25]
            }],
            "usage": {
                "prompt_tokens": 2,
                "total_tokens": 2
            }
        })))
        .mount(&server)
        .await;

    let provider = OpenAiAdapter::new(config(&server)).expect("provider");
    let response = provider
        .embed(ProviderEmbedRequest {
            model: "gpt-test".to_string(),
            input: "hello".to_string(),
            provider_options: Default::default(),
        })
        .await
        .expect("embeddings");

    assert_eq!(response.result.values, vec![0.125, -0.25]);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(2));
    assert_eq!(response.metadata.provider, ProviderKind::OpenAi);
}

#[tokio::test]
async fn maps_openai_chat_response() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "gpt-test",
            "messages": [{ "role": "user", "content": "hello" }]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-chat")
                .set_body_json(json!({
                    "id": "chatcmpl-test",
                    "model": "gpt-test",
                    "choices": [{
                        "message": { "role": "assistant", "content": "hi" },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2
                    }
                })),
        )
        .mount(&server)
        .await;

    let provider = OpenAiAdapter::new(config(&server)).expect("provider");
    let response = provider
        .chat(ProviderChatRequest {
            model: "gpt-test".to_string(),
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("chat");

    assert_eq!(response.result.text, "hi");
    assert_eq!(response.metadata.provider, ProviderKind::OpenAi);
    assert_eq!(response.metadata.request_id.as_deref(), Some("req-chat"));
}

#[tokio::test]
async fn sends_openai_client_request_id() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("x-client-request-id", "gateway-request-1"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id": "chatcmpl-test",
            "model": "gpt-test",
            "choices": [{
                "message": { "role": "assistant", "content": "hi" },
                "finish_reason": "stop"
            }]
        })))
        .mount(&server)
        .await;

    let provider = OpenAiAdapter::new(config(&server)).expect("provider");
    provider
        .chat_with_context(
            ProviderRequestContext {
                request_id: Some("gateway-request-1".to_string()),
            },
            ProviderChatRequest {
                model: "gpt-test".to_string(),
                messages: vec![ChatMessage::new(ChatRole::User, "hello")],
                options: ProviderGenerationOptions::default(),
                provider_options: Default::default(),
            },
        )
        .await
        .expect("chat");
}

#[tokio::test]
async fn maps_openai_completions_generate() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/completions"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "gpt-test",
            "prompt": "tell me",
            "max_tokens": 8,
            "temperature": 0.5,
            "stop": ["END"]
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-completion")
                .set_body_json(json!({
                    "id": "cmpl-test",
                    "object": "text_completion",
                    "model": "gpt-test",
                    "choices": [{
                        "text": "done",
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3
                    }
                })),
        )
        .mount(&server)
        .await;

    let provider = OpenAiAdapter::new(config(&server)).expect("provider");
    let response = provider
        .generate(ProviderGenerateRequest {
            model: "gpt-test".to_string(),
            prompt: "tell me".to_string(),
            options: ProviderGenerationOptions {
                max_tokens: Some(8),
                temperature: Some(0.5),
                stop: vec!["END".to_string()],
                ..ProviderGenerationOptions::default()
            },
            provider_options: Default::default(),
        })
        .await
        .expect("generate");

    assert_eq!(response.result.text, "done");
    assert_eq!(response.result.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(3));
    assert_eq!(response.metadata.response_id.as_deref(), Some("cmpl-test"));
    assert_eq!(
        response.metadata.request_id.as_deref(),
        Some("req-completion")
    );
}

#[tokio::test]
async fn streams_openai_completions_generate() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/completions"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "gpt-test",
            "prompt": "tell me",
            "stream": true
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-stream-completion")
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "data: {\"id\":\"cmpl-1\",\"model\":\"gpt-test\",\"choices\":[{\"text\":\"do\",\"finish_reason\":null}]}\n\n",
                    "data: {\"id\":\"cmpl-1\",\"model\":\"gpt-test\",\"choices\":[{\"text\":\"ne\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}\n\n",
                    "data: [DONE]\n\n"
                )),
        )
        .mount(&server)
        .await;

    let provider = OpenAiAdapter::new(config(&server)).expect("provider");
    let events = provider
        .stream_generate(ProviderGenerateRequest {
            model: "gpt-test".to_string(),
            prompt: "tell me".to_string(),
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect("stream generate")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<ProviderResult<Vec<_>>>()
        .expect("events");

    assert!(matches!(
        &events[0],
        ProviderStreamEvent::TokenBatch(batch) if batch.text == "do"
    ));
    assert!(matches!(
        &events[1],
        ProviderStreamEvent::Usage {
            usage: crate::providers::TokenUsage {
                total_tokens: Some(3),
                ..
            }
        }
    ));
    assert!(matches!(
        &events[2],
        ProviderStreamEvent::TokenBatch(batch) if batch.text == "ne"
    ));
    assert!(matches!(
        events[3],
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    ));
}

#[tokio::test]
async fn streams_openai_chat_chunks() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "gpt-test",
            "messages": [{ "role": "user", "content": "hello" }],
            "stream": true
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-stream")
                .insert_header("content-type", "text/event-stream")
                .set_body_string(concat!(
                    "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n",
                    "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                    "data: [DONE]\n\n"
                )),
        )
        .mount(&server)
        .await;

    let provider = OpenAiAdapter::new(config(&server)).expect("provider");
    let events = provider
        .stream_chat(ProviderChatRequest {
            model: "gpt-test".to_string(),
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
        ProviderStreamEvent::TokenBatch(batch)
            if batch.request_id == "req-stream" && batch.text == "hi"
    ));
    assert_eq!(
        events[1],
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    );
}

#[tokio::test]
async fn maps_openai_body_error_codes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/completions"))
        .and(header("authorization", "Bearer token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "error": {
                "message": "quota exceeded",
                "code": "insufficient_quota"
            }
        })))
        .mount(&server)
        .await;

    let provider = OpenAiAdapter::new(config(&server)).expect("provider");
    let err = provider
        .generate(ProviderGenerateRequest {
            model: "gpt-test".to_string(),
            prompt: "tell me".to_string(),
            options: ProviderGenerationOptions::default(),
            provider_options: Default::default(),
        })
        .await
        .expect_err("body error should fail");

    assert_eq!(err.kind, ProviderErrorKind::QuotaExceeded);
    assert_eq!(err.code.as_deref(), Some("insufficient_quota"));
}

#[test]
fn rejects_invalid_openai_completion_options() {
    let request = ProviderGenerateRequest {
        model: "gpt-test".to_string(),
        prompt: "tell me".to_string(),
        options: ProviderGenerationOptions {
            max_tokens: Some(0),
            ..ProviderGenerationOptions::default()
        },
        provider_options: Default::default(),
    };
    let err =
        openai_completion_body(&request, ProviderKind::OpenAi).expect_err("zero max_tokens fails");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    let request = ProviderGenerateRequest {
        model: "gpt-test".to_string(),
        prompt: "tell me".to_string(),
        options: ProviderGenerationOptions {
            top_p: Some(f32::INFINITY),
            ..ProviderGenerationOptions::default()
        },
        provider_options: Default::default(),
    };
    let err =
        openai_completion_body(&request, ProviderKind::OpenAi).expect_err("non-finite top_p fails");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[tokio::test]
async fn rejects_provider_options_colliding_with_completion_fields() {
    let provider = OpenAiAdapter::new(OpenAiAdapterConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        timeout: None,
    })
    .expect("provider");

    let err = provider
        .generate(ProviderGenerateRequest {
            model: "gpt-test".to_string(),
            prompt: "tell me".to_string(),
            options: ProviderGenerationOptions::default(),
            provider_options: [("prompt".to_string(), json!("override"))]
                .into_iter()
                .collect(),
        })
        .await
        .expect_err("provider_options prompt should be rejected");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[test]
fn completion_response_missing_text_is_invalid() {
    let body = json!({
        "id": "cmpl-1",
        "object": "text_completion",
        "model": "gpt-test",
        "choices": [{
            "finish_reason": "stop"
        }]
    });

    let err =
        openai_completion_response_from_body(Some("req-1".to_string()), body, ProviderKind::OpenAi)
            .expect_err("missing completion text should fail");

    assert_eq!(err.kind, ProviderErrorKind::Provider);
}
