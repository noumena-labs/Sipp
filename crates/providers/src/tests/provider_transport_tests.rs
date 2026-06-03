//! Tests the `provider_transport` module in `cogentlm-providers`.
//!
//! Covers provider transport construction, backend delegation, clone behavior,
//! and propagated backend errors with deterministic fake backends and no network
//! calls.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use cogentlm_core::{ChatMessage, ChatRole, FinishReason};
use futures_util::{stream, StreamExt};

use crate::{
    ProviderAuth, ProviderCapabilities, ProviderEmbeddingOutput, ProviderError, ProviderErrorKind,
    ProviderGenerationOptions, ProviderResponse, ProviderResponseMetadata, ProviderTextOutput,
    ProxyProtocol, SecretString,
};

use super::*;

#[derive(Default)]
struct FakeBackend {
    calls: Mutex<Vec<String>>,
}

impl FakeBackend {
    fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls lock").clone()
    }

    fn record(&self, call: impl Into<String>) {
        self.calls.lock().expect("calls lock").push(call.into());
    }
}

#[async_trait]
impl ProviderBackend for FakeBackend {
    fn kind(&self) -> ProviderKind {
        self.record("kind");
        ProviderKind::Proxy
    }

    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>> {
        self.record("list_models");
        Ok(vec![model("listed-model")])
    }

    async fn get_model(&self, model_id: &str) -> ProviderResult<ProviderModel> {
        self.record(format!("get_model:{model_id}"));
        if model_id == "missing" {
            return Err(ProviderError::new(
                ProviderErrorKind::ModelNotFound,
                ProviderKind::Proxy,
                "missing model",
            ));
        }
        Ok(model(model_id))
    }

    async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse> {
        self.record(format!("chat:{}", req.model));
        Ok(text_response(req.model, "chat text"))
    }

    async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        self.record(format!("generate:{}", req.model));
        Ok(text_response(req.model, "generated text"))
    }

    async fn embed(&self, req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse> {
        self.record(format!("embed:{}", req.model));
        Ok(ProviderResponse {
            result: ProviderEmbeddingOutput {
                values: vec![0.25, -0.5],
            },
            usage: None,
            metadata: metadata(ProviderKind::Proxy, req.model),
        })
    }

    async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.record(format!("stream_chat:{}", req.model));
        Ok(Box::pin(stream::iter(vec![Ok(
            ProviderStreamEvent::Finished {
                finish_reason: FinishReason::Stop,
            },
        )])))
    }
}

#[tokio::test]
async fn transport_delegates_every_backend_method_and_clone_shares_backend() {
    let backend = Arc::new(FakeBackend::default());
    let transport = ProviderTransport::from_backend(backend.clone());
    let cloned = transport.clone();

    assert_eq!(transport.kind(), ProviderKind::Proxy);
    assert_eq!(
        cloned.list_models().await.expect("models")[0].id,
        "listed-model"
    );
    assert_eq!(
        transport.get_model("model-a").await.expect("model").id,
        "model-a"
    );

    let chat = cloned.chat(chat_request("chat-model")).await.expect("chat");
    assert_eq!(chat.result.text, "chat text");

    let generated = transport
        .generate(generate_request("generate-model"))
        .await
        .expect("generate");
    assert_eq!(generated.result.text, "generated text");

    let embedding = cloned
        .embed(embed_request("embed-model"))
        .await
        .expect("embedding");
    assert_eq!(embedding.result.values, vec![0.25, -0.5]);

    let stream_event = transport
        .stream_chat(chat_request("stream-model"))
        .await
        .expect("stream")
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .next()
        .expect("stream event")
        .expect("stream result");
    assert_eq!(
        stream_event,
        ProviderStreamEvent::Finished {
            finish_reason: FinishReason::Stop
        }
    );

    assert_eq!(
        backend.calls(),
        vec![
            "kind",
            "list_models",
            "get_model:model-a",
            "chat:chat-model",
            "generate:generate-model",
            "embed:embed-model",
            "stream_chat:stream-model"
        ]
    );
}

#[tokio::test]
async fn transport_preserves_backend_errors() {
    let transport = ProviderTransport::from_backend(Arc::new(FakeBackend::default()));

    let err = transport
        .get_model("missing")
        .await
        .expect_err("missing model should propagate");

    assert_eq!(err.kind, ProviderErrorKind::ModelNotFound);
    assert_eq!(err.provider, ProviderKind::Proxy);
}

#[test]
fn transport_provider_constructors_build_local_backends() {
    let proxy = ProviderTransport::proxy(ProxyConfig {
        base_url: "http://localhost".to_string(),
        auth: ProviderAuth::Bearer(SecretString::new("token")),
        protocol: ProxyProtocol::OpenAiCompatible,
        static_headers: Vec::new(),
        timeout: None,
    })
    .expect("proxy transport");
    assert_eq!(proxy.kind(), ProviderKind::Proxy);

    let openai = ProviderTransport::openai(OpenAiConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        timeout: None,
    })
    .expect("openai transport");
    assert_eq!(openai.kind(), ProviderKind::OpenAi);

    let anthropic = ProviderTransport::anthropic(AnthropicConfig {
        api_key: SecretString::new("token"),
        base_url: Some("http://localhost".to_string()),
        version: None,
        timeout: None,
    })
    .expect("anthropic transport");
    assert_eq!(anthropic.kind(), ProviderKind::Anthropic);
}

fn model(id: &str) -> ProviderModel {
    ProviderModel {
        id: id.to_string(),
        provider: ProviderKind::Proxy,
        display_name: None,
        capabilities: ProviderCapabilities::unknown(),
        context_window: None,
        max_output_tokens: None,
        raw: serde_json::json!({ "id": id }),
    }
}

fn text_response(model: String, text: &str) -> ProviderChatResponse {
    ProviderResponse {
        result: ProviderTextOutput {
            text: text.to_string(),
            finish_reason: FinishReason::Stop,
        },
        usage: None,
        metadata: metadata(ProviderKind::Proxy, model),
    }
}

fn metadata(provider: ProviderKind, model: String) -> ProviderResponseMetadata {
    ProviderResponseMetadata {
        provider,
        model,
        request_id: None,
        response_id: None,
        finish_reason_raw: None,
        raw: serde_json::Value::Null,
    }
}

fn chat_request(model: &str) -> ProviderChatRequest {
    ProviderChatRequest {
        model: model.to_string(),
        messages: vec![ChatMessage::new(ChatRole::User, "hello")],
        options: ProviderGenerationOptions::default(),
        provider_options: Default::default(),
    }
}

fn generate_request(model: &str) -> ProviderGenerateRequest {
    ProviderGenerateRequest {
        model: model.to_string(),
        prompt: "tell me".to_string(),
        options: ProviderGenerationOptions::default(),
        provider_options: Default::default(),
    }
}

fn embed_request(model: &str) -> ProviderEmbedRequest {
    ProviderEmbedRequest {
        model: model.to_string(),
        input: "hello".to_string(),
        provider_options: Default::default(),
    }
}
