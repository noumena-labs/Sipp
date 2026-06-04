//! Backend tests for the parent module.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use axum::response::IntoResponse;
use cogentlm_core::{ChatMessage, ChatRole, FinishReason, TokenBatch};
use cogentlm_gateway_providers::{
    GatewayBackendAdapter, ProviderChatRequest, ProviderChatResponse, ProviderEmbedRequest,
    ProviderEmbeddingResponse, ProviderError, ProviderErrorKind, ProviderGenerateRequest,
    ProviderGenerateResponse, ProviderKind, ProviderModel, ProviderResult, ProviderStream,
    ProviderStreamEvent, TokenUsage,
};
use futures_util::{stream, StreamExt};
use serde_json::json;

use super::*;

#[tokio::test]
async fn provider_backend_stream_query_uses_generate_stream() {
    let captured = Arc::new(Mutex::new(None));
    let adapter = Arc::new(RecordingAdapter {
        captured: captured.clone(),
    });
    let backend = ProviderGatewayBackend::from_provider_backend("private-model", adapter)
        .expect("provider backend");

    let events = backend
        .stream_query(BackendQueryRequest {
            prompt: "raw prompt".to_string(),
            options: BackendGenerationOptions {
                max_tokens: Some(8),
                temperature: Some(0.5),
                top_p: Some(0.9),
                stop: vec!["END".to_string()],
            },
            gateway_options: Default::default(),
        })
        .await
        .expect("stream query")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<GatewayResult<Vec<_>>>()
        .expect("events");

    let request = captured
        .lock()
        .expect("captured request lock")
        .take()
        .expect("captured request");
    assert_eq!(request.model, "private-model");
    assert_eq!(request.prompt, "raw prompt");
    assert_eq!(request.options.max_tokens, Some(8));
    assert_eq!(request.options.stop, vec!["END"]);
    assert!(request.provider_options.is_empty());

    assert!(matches!(
        &events[0],
        GatewayStreamEvent::TokenBatch(TokenBatch { text, .. }) if text == "token"
    ));
    assert!(matches!(
        &events[1],
        GatewayStreamEvent::Usage {
            usage: TokenUsage {
                total_tokens: Some(2),
                ..
            }
        }
    ));
    assert!(matches!(
        events[2],
        GatewayStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    ));
}

#[tokio::test]
async fn provider_backend_rejects_request_gateway_options() {
    let backend = ProviderGatewayBackend::from_provider_backend(
        "private-model",
        Arc::new(StreamErrorAdapter),
    )
    .expect("provider backend");
    let gateway_options = [("store".to_string(), json!(true))].into_iter().collect();

    let query_error = backend
        .query(BackendQueryRequest {
            prompt: "raw prompt".to_string(),
            options: BackendGenerationOptions::default(),
            gateway_options,
        })
        .await
        .expect_err("provider query should reject request gateway_options");

    assert_eq!(query_error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        query_error.message,
        "provider gateway backend does not accept request gateway_options"
    );

    let chat_error = backend
        .chat(BackendChatRequest {
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: BackendGenerationOptions::default(),
            gateway_options: [("metadata".to_string(), json!({ "tenant": "a" }))]
                .into_iter()
                .collect(),
        })
        .await
        .expect_err("provider chat should reject request gateway_options");
    assert_eq!(chat_error.kind, GatewayErrorKind::InvalidRequest);

    let embed_error = backend
        .embed(BackendEmbedRequest {
            input: "hello".to_string(),
            gateway_options: [("encoding_format".to_string(), json!("base64"))]
                .into_iter()
                .collect(),
        })
        .await
        .expect_err("provider embed should reject request gateway_options");
    assert_eq!(embed_error.kind, GatewayErrorKind::InvalidRequest);
}

#[test]
fn provider_backend_rejects_invalid_private_model_names() {
    for (model, message) in [
        (" \t ", "provider backend model must not be empty"),
        (
            " private-model ",
            "provider backend model must not contain surrounding whitespace",
        ),
    ] {
        let error = match ProviderGatewayBackend::from_provider_backend(
            model,
            Arc::new(StreamErrorAdapter),
        ) {
            Ok(_) => panic!("invalid private model must fail"),
            Err(error) => error,
        };
        assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
        assert_eq!(error.message, message);
    }
}

#[tokio::test]
async fn mock_backend_rejects_request_gateway_options() {
    let backend = MockBackend::new("mock:", 3);

    let query_error = backend
        .query(BackendQueryRequest {
            prompt: "raw prompt".to_string(),
            options: BackendGenerationOptions::default(),
            gateway_options: [("debug".to_string(), json!(true))].into_iter().collect(),
        })
        .await
        .expect_err("mock query should reject request gateway_options");

    assert_eq!(query_error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        query_error.message,
        "mock gateway backend does not accept gateway_options"
    );

    let stream_query_error = backend
        .stream_query(BackendQueryRequest {
            prompt: "raw prompt".to_string(),
            options: BackendGenerationOptions::default(),
            gateway_options: [("debug".to_string(), json!(true))].into_iter().collect(),
        })
        .await
        .err()
        .expect("mock stream query should reject request gateway_options");
    assert_eq!(stream_query_error.kind, GatewayErrorKind::InvalidRequest);

    let chat_error = backend
        .chat(BackendChatRequest {
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: BackendGenerationOptions::default(),
            gateway_options: [("metadata".to_string(), json!({ "tenant": "a" }))]
                .into_iter()
                .collect(),
        })
        .await
        .expect_err("mock chat should reject request gateway_options");
    assert_eq!(chat_error.kind, GatewayErrorKind::InvalidRequest);

    let stream_chat_error = backend
        .stream_chat(BackendChatRequest {
            messages: vec![ChatMessage::new(ChatRole::User, "hello")],
            options: BackendGenerationOptions::default(),
            gateway_options: [("metadata".to_string(), json!({ "tenant": "a" }))]
                .into_iter()
                .collect(),
        })
        .await
        .err()
        .expect("mock stream chat should reject request gateway_options");
    assert_eq!(stream_chat_error.kind, GatewayErrorKind::InvalidRequest);

    let embed_error = backend
        .embed(BackendEmbedRequest {
            input: "hello".to_string(),
            gateway_options: [("encoding_format".to_string(), json!("base64"))]
                .into_iter()
                .collect(),
        })
        .await
        .expect_err("mock embed should reject request gateway_options");
    assert_eq!(embed_error.kind, GatewayErrorKind::InvalidRequest);
}

#[test]
fn provider_errors_are_normalized_before_gateway_boundary() {
    let error = ProviderError {
        kind: ProviderErrorKind::RateLimited,
        provider: ProviderKind::OpenAiCompatible,
        status: Some(429),
        code: Some("provider-secret-code".to_string()),
        message: "provider rejected provider-secret-token".to_string(),
        retry_after: Some(Duration::from_millis(1500)),
        request_id: Some("req-provider-secret-token".to_string()),
        raw: Some(Box::new(json!({
            "error": {
                "message": "provider-secret-token",
                "code": "provider-secret-code"
            }
        }))),
    };

    let gateway_error = provider_error(error);

    assert_eq!(gateway_error.kind, GatewayErrorKind::RateLimited);
    assert_eq!(gateway_error.code(), "rate_limited");
    assert_eq!(gateway_error.message, "provider rate limit exceeded");
    assert_eq!(gateway_error.retry_after, Some(Duration::from_millis(1500)));
    assert!(!gateway_error.to_string().contains("provider-secret"));
    assert!(!format!("{gateway_error:?}").contains("provider-secret"));

    let response = gateway_error.into_response();
    assert!(response.headers().get("x-request-id").is_none());
}

#[test]
fn local_engine_errors_are_normalized_before_gateway_boundary() {
    let cases = [
        (
            cogentlm_engine::Error::ModelLoad {
                path: r"C:\gateway-secret\models\private.gguf".to_string(),
            },
            GatewayErrorKind::ModelNotFound,
            "local CogentEngine model was not found",
        ),
        (
            cogentlm_engine::Error::RuntimeCommand(
                "runtime command included gateway-secret".to_string(),
            ),
            GatewayErrorKind::Overloaded,
            "local CogentEngine runtime is overloaded",
        ),
        (
            cogentlm_engine::Error::UnsupportedOperation {
                operation: "embed",
                reason: "private model class included gateway-secret".to_string(),
            },
            GatewayErrorKind::UnsupportedFeature,
            "local CogentEngine backend does not support this operation",
        ),
        (
            cogentlm_engine::Error::InvalidConfig("gateway-secret config"),
            GatewayErrorKind::InvalidRequest,
            "local CogentEngine configuration is invalid",
        ),
        (
            cogentlm_engine::Error::InvalidRequest("gateway-secret request"),
            GatewayErrorKind::InvalidRequest,
            "local CogentEngine request is invalid",
        ),
    ];

    for (engine, kind, message) in cases {
        let gateway_error = engine_error(engine);

        assert_eq!(gateway_error.kind, kind);
        assert_eq!(gateway_error.message, message);
        assert!(!gateway_error.to_string().contains("gateway-secret"));
        assert!(!format!("{gateway_error:?}").contains("gateway-secret"));
    }
}

#[tokio::test]
async fn provider_stream_errors_are_normalized_before_gateway_boundary() {
    let backend = ProviderGatewayBackend::from_provider_backend(
        "private-model",
        Arc::new(StreamErrorAdapter),
    )
    .expect("provider backend");

    let mut stream = backend
        .stream_query(BackendQueryRequest {
            prompt: "raw prompt".to_string(),
            options: BackendGenerationOptions::default(),
            gateway_options: Default::default(),
        })
        .await
        .expect("stream query");

    let error = stream
        .next()
        .await
        .expect("stream item")
        .expect_err("provider stream error should fail");

    assert_eq!(error.kind, GatewayErrorKind::RateLimited);
    assert_eq!(error.code(), "rate_limited");
    assert_eq!(error.message, "provider rate limit exceeded");
    assert_eq!(error.retry_after, Some(Duration::from_millis(1500)));
    assert!(!error.to_string().contains("provider-secret"));
    assert!(!format!("{error:?}").contains("provider-secret"));
}

struct RecordingAdapter {
    captured: Arc<Mutex<Option<ProviderGenerateRequest>>>,
}

#[async_trait]
impl GatewayBackendAdapter for RecordingAdapter {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAiCompatible
    }

    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>> {
        Ok(Vec::new())
    }

    async fn get_model(&self, _model: &str) -> ProviderResult<ProviderModel> {
        Err(unsupported("get_model"))
    }

    async fn chat(&self, _req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse> {
        Err(unsupported("chat"))
    }

    async fn generate(
        &self,
        _req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        Err(unsupported("generate"))
    }

    async fn stream_generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        *self.captured.lock().expect("captured request lock") = Some(req);
        Ok(Box::pin(stream::iter([
            Ok(ProviderStreamEvent::TokenBatch(TokenBatch {
                request_id: "req-1".to_string(),
                stream_id: 0,
                sequence_start: 0,
                text: "token".to_string(),
                frame_count: 1,
                byte_count: 5,
                stats: cogentlm_core::TokenEmissionStats {
                    frames_sent: 1,
                    bytes_sent: 5,
                    batches_sent: 1,
                },
            })),
            Ok(ProviderStreamEvent::Usage {
                usage: TokenUsage {
                    input_tokens: Some(1),
                    output_tokens: Some(1),
                    total_tokens: Some(2),
                },
            }),
            Ok(ProviderStreamEvent::Finished {
                finish_reason: FinishReason::Stop,
            }),
        ])))
    }

    async fn embed(&self, _req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse> {
        Err(unsupported("embed"))
    }

    async fn stream_chat(
        &self,
        _req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        Err(unsupported("stream_chat"))
    }
}

struct StreamErrorAdapter;

#[async_trait]
impl GatewayBackendAdapter for StreamErrorAdapter {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAiCompatible
    }

    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>> {
        Ok(Vec::new())
    }

    async fn get_model(&self, _model: &str) -> ProviderResult<ProviderModel> {
        Err(unsupported("get_model"))
    }

    async fn chat(&self, _req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse> {
        Err(unsupported("chat"))
    }

    async fn generate(
        &self,
        _req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        Err(unsupported("generate"))
    }

    async fn stream_generate(
        &self,
        _req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        Ok(Box::pin(stream::iter([Err(provider_stream_error())])))
    }

    async fn embed(&self, _req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse> {
        Err(unsupported("embed"))
    }

    async fn stream_chat(
        &self,
        _req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        Ok(Box::pin(stream::iter([Err(provider_stream_error())])))
    }
}

fn unsupported(operation: &'static str) -> ProviderError {
    ProviderError::new(
        ProviderErrorKind::UnsupportedFeature,
        ProviderKind::OpenAiCompatible,
        format!("{operation} is not supported by the recording adapter"),
    )
}

fn provider_stream_error() -> ProviderError {
    ProviderError {
        kind: ProviderErrorKind::RateLimited,
        provider: ProviderKind::OpenAiCompatible,
        status: Some(429),
        code: Some("provider-secret-code".to_string()),
        message: "provider rejected provider-secret-token".to_string(),
        retry_after: Some(Duration::from_millis(1500)),
        request_id: Some("req-provider-secret-token".to_string()),
        raw: Some(Box::new(json!({
            "error": {
                "message": "provider-secret-token",
                "code": "provider-secret-code"
            }
        }))),
    }
}
