//! Tests the `remote_endpoint` module in `cogentlm-client`.
//!
//! Covers endpoint resolution, remote configuration, facade validation, and run wrappers with deterministic fakes rather than a live local engine.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use cogentlm_core::{ChatMessage, ChatRole, FinishReason, TokenBatch, TokenEmissionStats};
use cogentlm_providers::{
    ProviderBackend, ProviderChatRequest, ProviderChatResponse, ProviderEmbedRequest,
    ProviderEmbeddingOutput, ProviderEmbeddingResponse, ProviderError, ProviderErrorKind,
    ProviderGenerateRequest, ProviderGenerateResponse, ProviderKind, ProviderModel,
    ProviderResponse, ProviderResponseMetadata, ProviderStream, ProviderStreamEvent,
    ProviderTextOutput, ProviderTransport, TokenUsage,
};
use futures::executor::block_on;
use futures::stream;
use futures::StreamExt;
use serde_json::json;

use super::*;
use crate::{CogentTextOptions, LocalTextOptions};

#[derive(Default)]
struct FakeBackend {
    calls: Mutex<Vec<&'static str>>,
}

#[async_trait]
impl ProviderBackend for FakeBackend {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Proxy
    }

    async fn list_models(&self) -> cogentlm_providers::ProviderResult<Vec<ProviderModel>> {
        Err(unused_call("list_models"))
    }

    async fn get_model(&self, _model: &str) -> cogentlm_providers::ProviderResult<ProviderModel> {
        Err(unused_call("get_model"))
    }

    async fn chat(
        &self,
        req: ProviderChatRequest,
    ) -> cogentlm_providers::ProviderResult<ProviderChatResponse> {
        self.calls.lock().expect("calls").push("chat");
        Ok(text_response(
            &req.model,
            req.messages
                .first()
                .map(|message| message.content.as_str())
                .unwrap_or_default(),
        ))
    }

    async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> cogentlm_providers::ProviderResult<ProviderGenerateResponse> {
        self.calls.lock().expect("calls").push("generate");
        Ok(text_response(&req.model, &req.prompt))
    }

    async fn embed(
        &self,
        req: ProviderEmbedRequest,
    ) -> cogentlm_providers::ProviderResult<ProviderEmbeddingResponse> {
        self.calls.lock().expect("calls").push("embed");
        Ok(ProviderResponse {
            result: ProviderEmbeddingOutput {
                values: vec![1.0, 2.0, 3.0],
            },
            usage: Some(TokenUsage {
                input_tokens: Some(req.input.len() as u32),
                output_tokens: None,
                total_tokens: Some(req.input.len() as u32),
            }),
            metadata: metadata(&req.model),
        })
    }

    async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> cogentlm_providers::ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.calls.lock().expect("calls").push("stream_chat");
        let events = vec![
            Ok(ProviderStreamEvent::TokenBatch(token_batch("a"))),
            Ok(ProviderStreamEvent::TokenBatch(token_batch("b"))),
            Ok(ProviderStreamEvent::Usage {
                usage: TokenUsage {
                    input_tokens: Some(2),
                    output_tokens: Some(2),
                    total_tokens: Some(4),
                },
            }),
            Ok(ProviderStreamEvent::Finished {
                finish_reason: FinishReason::Length,
            }),
        ];
        assert_eq!(req.model, "remote-model");
        Ok(Box::pin(stream::iter(events)))
    }
}

fn unused_call(name: &'static str) -> ProviderError {
    ProviderError::new(
        ProviderErrorKind::UnsupportedFeature,
        ProviderKind::Proxy,
        format!("{name} is not used by this test"),
    )
}

fn metadata(model: &str) -> ProviderResponseMetadata {
    ProviderResponseMetadata {
        provider: ProviderKind::Proxy,
        model: model.to_string(),
        request_id: Some("req-1".to_string()),
        response_id: Some("resp-1".to_string()),
        finish_reason_raw: None,
        raw: json!({}),
    }
}

fn text_response(model: &str, text: &str) -> ProviderGenerateResponse {
    ProviderResponse {
        result: ProviderTextOutput {
            text: format!("echo:{text}"),
            finish_reason: FinishReason::Stop,
        },
        usage: Some(TokenUsage {
            input_tokens: Some(1),
            output_tokens: Some(1),
            total_tokens: Some(2),
        }),
        metadata: metadata(model),
    }
}

fn token_batch(text: &str) -> TokenBatch {
    TokenBatch {
        request_id: "req-1".to_string(),
        stream_id: 7,
        sequence_start: 0,
        text: text.to_string(),
        frame_count: 1,
        byte_count: text.len() as u32,
        stats: TokenEmissionStats {
            frames_sent: 1,
            bytes_sent: text.len() as u64,
            batches_sent: 1,
        },
    }
}

fn endpoint(backend: Arc<FakeBackend>) -> RemoteEndpoint {
    RemoteEndpoint::new(
        EndpointRef::Remote {
            id: "remote".to_string(),
        },
        "remote-model".to_string(),
        EndpointCapabilities::unknown(),
        ProviderTransport::from_backend(backend),
        RemoteExecutor::new().expect("remote executor"),
    )
}

#[test]
fn remote_generation_options_preserve_common_text_fields() {
    let options = remote_generation_options(CogentTextOptions {
        max_tokens: Some(7),
        temperature: Some(0.25),
        top_p: Some(0.9),
        stop: vec!["stop".to_string()],
    });

    assert_eq!(options.max_tokens, Some(7));
    assert_eq!(options.temperature, Some(0.25));
    assert_eq!(options.top_p, Some(0.9));
    assert_eq!(options.stop, vec!["stop"]);
}

#[test]
fn query_maps_provider_response_to_client_response() {
    let backend = Arc::new(FakeBackend::default());
    let endpoint = endpoint(Arc::clone(&backend));
    let response = block_on(endpoint.query(CogentQueryRequest {
        prompt: "hello".to_string(),
        options: CogentTextOptions {
            max_tokens: Some(3),
            ..CogentTextOptions::default()
        },
        ..CogentQueryRequest::default()
    }))
    .expect("query response");

    assert_eq!(response.text, "echo:hello");
    assert_eq!(response.finish_reason, FinishReason::Stop);
    assert_eq!(response.endpoint, *endpoint.endpoint());
    assert_eq!(
        backend.calls.lock().expect("calls").as_slice(),
        &["generate"]
    );
}

#[test]
fn query_rejects_token_emission_before_transport_call() {
    let backend = Arc::new(FakeBackend::default());
    let endpoint = endpoint(Arc::clone(&backend));
    let error = block_on(endpoint.query(CogentQueryRequest {
        emit_tokens: true,
        ..CogentQueryRequest::default()
    }))
    .expect_err("query token emission is unsupported");

    assert!(matches!(
        error,
        CogentError::UnsupportedOperation {
            operation: "query",
            ..
        }
    ));
    assert!(backend.calls.lock().expect("calls").is_empty());
}

#[test]
fn chat_stream_forwards_token_batches_and_final_response() {
    let backend = Arc::new(FakeBackend::default());
    let endpoint = endpoint(Arc::clone(&backend));
    let run = endpoint.chat(CogentChatRequest {
        messages: vec![ChatMessage::new(ChatRole::User, "hello")],
        emit_tokens: true,
        ..CogentChatRequest::default()
    });
    let (tokens, response) = run.into_parts();
    let (response, tokens) = block_on(async {
        let response = response.await.expect("chat response");
        let tokens = tokens.collect::<Vec<_>>().await;
        (response, tokens)
    });

    assert_eq!(response.text, "ab");
    assert_eq!(response.finish_reason, FinishReason::Length);
    assert_eq!(response.usage.expect("usage").total_tokens, Some(4));
    assert_eq!(
        tokens
            .iter()
            .map(|batch| batch.text.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    assert_eq!(
        backend.calls.lock().expect("calls").as_slice(),
        &["stream_chat"]
    );
}

#[test]
fn chat_rejects_local_options_before_transport_call() {
    let backend = Arc::new(FakeBackend::default());
    let endpoint = endpoint(Arc::clone(&backend));
    let error = block_on(endpoint.chat(CogentChatRequest {
        local: LocalTextOptions {
            context_key: Some("ctx".to_string()),
            ..LocalTextOptions::default()
        },
        ..CogentChatRequest::default()
    }))
    .expect_err("local options are invalid for remote chat");

    assert!(matches!(error, CogentError::InvalidRequest(_)));
    assert!(backend.calls.lock().expect("calls").is_empty());
}

#[test]
fn embed_maps_provider_response_to_client_response() {
    let backend = Arc::new(FakeBackend::default());
    let endpoint = endpoint(Arc::clone(&backend));
    let response = block_on(endpoint.embed(CogentEmbedRequest {
        input: "abc".to_string(),
        ..CogentEmbedRequest::default()
    }))
    .expect("embed response");

    assert_eq!(response.values, vec![1.0, 2.0, 3.0]);
    assert_eq!(response.endpoint, *endpoint.endpoint());
    assert_eq!(response.usage.expect("usage").input_tokens, Some(3));
    assert_eq!(backend.calls.lock().expect("calls").as_slice(), &["embed"]);
}
