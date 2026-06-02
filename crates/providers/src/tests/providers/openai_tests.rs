//! Tests the `providers::openai` module in `cogentlm-providers`.
//!
//! Covers provider request mapping, response parsing, transport, and stream behavior with deterministic local fixtures and no live network calls.

use std::time::Duration;

use cogentlm_core::{ChatMessage, ChatRole, FinishReason};
use serde_json::json;
use wiremock::matchers::{body_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

use super::*;
use crate::{ProviderGenerationOptions, SecretString};

fn config(server: &MockServer) -> OpenAiConfig {
    OpenAiConfig {
        api_key: SecretString::new("token"),
        base_url: Some(server.uri()),
        timeout: None,
    }
}

#[test]
fn rejects_zero_timeout() {
    let mut config = OpenAiConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        timeout: Some(Duration::ZERO),
    };

    let err = match OpenAiProvider::new(config.clone()) {
        Ok(_) => panic!("zero timeout should be rejected"),
        Err(err) => err,
    };
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

    config.timeout = Some(Duration::from_millis(1));
    OpenAiProvider::new(config).expect("positive timeout");
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

    let provider = OpenAiProvider::new(config(&server)).expect("provider");
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

    let provider = OpenAiProvider::new(config(&server)).expect("provider");
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

    let provider = OpenAiProvider::new(config(&server)).expect("provider");
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

    let provider = OpenAiProvider::new(config(&server)).expect("provider");
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
async fn maps_openai_responses_generate() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer token"))
        .and(body_json(json!({
            "model": "gpt-test",
            "input": "tell me",
            "max_output_tokens": 8,
            "temperature": 0.5
        })))
        .respond_with(
            ResponseTemplate::new(200)
                .insert_header("x-request-id", "req-response")
                .set_body_json(json!({
                    "id": "resp-test",
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-test",
                    "output": [{
                        "type": "message",
                        "content": [{
                            "type": "output_text",
                            "text": "done"
                        }]
                    }],
                    "usage": {
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3
                    }
                })),
        )
        .mount(&server)
        .await;

    let provider = OpenAiProvider::new(config(&server)).expect("provider");
    let response = provider
        .generate(ProviderGenerateRequest {
            model: "gpt-test".to_string(),
            prompt: "tell me".to_string(),
            options: ProviderGenerationOptions {
                max_tokens: Some(8),
                temperature: Some(0.5),
                ..ProviderGenerationOptions::default()
            },
            provider_options: Default::default(),
        })
        .await
        .expect("generate");

    assert_eq!(response.result.text, "done");
    assert_eq!(response.result.finish_reason, FinishReason::Stop);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(3));
    assert_eq!(response.metadata.response_id.as_deref(), Some("resp-test"));
}

#[tokio::test]
async fn maps_openai_body_error_codes() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/responses"))
        .and(header("authorization", "Bearer token"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "error": {
                "message": "quota exceeded",
                "code": "insufficient_quota"
            }
        })))
        .mount(&server)
        .await;

    let provider = OpenAiProvider::new(config(&server)).expect("provider");
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

#[tokio::test]
async fn rejects_unmapped_openai_responses_stop() {
    let provider = OpenAiProvider::new(OpenAiConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        timeout: None,
    })
    .expect("provider");

    let err = provider
        .generate(ProviderGenerateRequest {
            model: "gpt-test".to_string(),
            prompt: "tell me".to_string(),
            options: ProviderGenerationOptions {
                stop: vec!["END".to_string()],
                ..ProviderGenerationOptions::default()
            },
            provider_options: Default::default(),
        })
        .await
        .expect_err("stop should be rejected");

    assert_eq!(err.kind, ProviderErrorKind::UnsupportedFeature);
}

#[test]
fn rejects_invalid_openai_responses_options() {
    let request = ProviderGenerateRequest {
        model: "gpt-test".to_string(),
        prompt: "tell me".to_string(),
        options: ProviderGenerationOptions {
            max_tokens: Some(0),
            ..ProviderGenerationOptions::default()
        },
        provider_options: Default::default(),
    };
    let err = openai_responses_body(&request).expect_err("zero max_tokens fails");
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
    let err = openai_responses_body(&request).expect_err("non-finite top_p fails");
    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[tokio::test]
async fn rejects_provider_options_colliding_with_responses_fields() {
    let provider = OpenAiProvider::new(OpenAiConfig {
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
            provider_options: [("stop".to_string(), json!(["END"]))].into_iter().collect(),
        })
        .await
        .expect_err("provider_options stop should be rejected");

    assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);
}

#[test]
fn responses_tool_call_output_yields_empty_text() {
    let body = json!({
        "id": "resp-1",
        "object": "response",
        "status": "completed",
        "model": "gpt-test",
        "output": [{
            "type": "function_call",
            "name": "get_weather",
            "arguments": "{}",
            "call_id": "call_1"
        }]
    });

    let response = openai_responses_response_from_body(Some("req-1".to_string()), body)
        .expect("tool-call responses output should parse");

    assert_eq!(response.result.text, "");
    assert!(response.metadata.raw.pointer("/output/0/type").is_some());
}
