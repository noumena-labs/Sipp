use async_trait::async_trait;

use super::openai_compat::{
    openai_chat_body, openai_chat_response_from_body, openai_embedding_body,
    openai_embedding_response_from_body, openai_model_from_value, openai_models_from_body,
    openai_stream_chat_body, openai_stream_events,
};
use crate::{
    HttpTransport, ProviderBackend, ProviderChatRequest, ProviderChatResponse,
    ProviderEmbedRequest, ProviderEmbeddingResponse, ProviderError, ProviderErrorKind,
    ProviderGenerateRequest, ProviderGenerateResponse, ProviderKind, ProviderModel, ProviderResult,
    ProviderStream, ProviderStreamEvent, ProxyConfig, ProxyProtocol,
};

pub struct ProxyProvider {
    transport: HttpTransport,
}

impl ProxyProvider {
    pub fn new(config: ProxyConfig) -> ProviderResult<Self> {
        let ProxyConfig {
            base_url,
            auth,
            protocol,
            static_headers,
            timeout,
        } = config;
        match protocol {
            ProxyProtocol::OpenAiCompatible => {}
        }

        let transport = HttpTransport::new_with_options(
            ProviderKind::Proxy,
            base_url,
            auth,
            static_headers,
            timeout,
        )?;
        Ok(Self { transport })
    }
}

#[async_trait]
impl ProviderBackend for ProxyProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::Proxy
    }

    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>> {
        let response = self.transport.get_json("/models").await?;
        openai_models_from_body(&response.body, ProviderKind::Proxy)
    }

    async fn get_model(&self, model: &str) -> ProviderResult<ProviderModel> {
        let response = self.transport.get_json(&format!("/models/{model}")).await?;
        openai_model_from_value(&response.body, ProviderKind::Proxy)
    }

    async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse> {
        let body = openai_chat_body(&req, ProviderKind::Proxy)?;
        let response = self.transport.post_json("/chat/completions", &body).await?;
        openai_chat_response_from_body(response.request_id, response.body, ProviderKind::Proxy)
    }

    async fn generate(
        &self,
        _req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        Err(ProviderError::new(
            ProviderErrorKind::UnsupportedFeature,
            ProviderKind::Proxy,
            "generate is not supported by the proxy provider yet",
        ))
    }

    async fn embed(&self, req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse> {
        let body = openai_embedding_body(&req, ProviderKind::Proxy)?;
        let response = self.transport.post_json("/embeddings", &body).await?;
        openai_embedding_response_from_body(response.request_id, response.body, ProviderKind::Proxy)
    }

    async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        let body = openai_stream_chat_body(&req, ProviderKind::Proxy)?;
        let response = self
            .transport
            .post_json_stream("/chat/completions", &body)
            .await?;
        Ok(openai_stream_events(
            response.request_id,
            response.stream,
            ProviderKind::Proxy,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use cogentlm_core::{ChatMessage, ChatRole, FinishReason, TokenBatch};
    use futures_util::StreamExt;
    use serde_json::json;
    use wiremock::matchers::{body_json, header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::{ProviderAuth, ProviderGenerationOptions, ProxyProtocol, SecretString, TokenUsage};

    fn config(server: &MockServer) -> ProxyConfig {
        ProxyConfig {
            base_url: server.uri(),
            auth: ProviderAuth::Bearer(SecretString::new("token")),
            protocol: ProxyProtocol::OpenAiCompatible,
            static_headers: Vec::new(),
            timeout: None,
        }
    }

    #[test]
    fn rejects_zero_timeout() {
        let mut config = ProxyConfig {
            base_url: "http://localhost".to_string(),
            auth: ProviderAuth::Bearer(SecretString::new("token")),
            protocol: ProxyProtocol::OpenAiCompatible,
            static_headers: Vec::new(),
            timeout: Some(Duration::ZERO),
        };

        let err = match ProxyProvider::new(config.clone()) {
            Ok(_) => panic!("zero timeout should be rejected"),
            Err(err) => err,
        };
        assert_eq!(err.kind, ProviderErrorKind::InvalidRequest);

        config.timeout = Some(Duration::from_millis(1));
        ProxyProvider::new(config).expect("positive timeout");
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
        let provider = ProxyProvider::new(config).expect("proxy provider");
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

        let provider = ProxyProvider::new(config(&server)).expect("proxy provider");
        let model = provider.get_model("model-a").await.expect("model");

        assert_eq!(model.id, "model-a");
        assert_eq!(model.provider, ProviderKind::Proxy);
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

        let provider = ProxyProvider::new(config(&server)).expect("proxy provider");
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
        assert_eq!(response.metadata.provider, ProviderKind::Proxy);
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

        let provider = ProxyProvider::new(config(&server)).expect("proxy provider");
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
    async fn rejects_provider_options_colliding_with_typed_fields() {
        let server = MockServer::start().await;
        let provider = ProxyProvider::new(config(&server)).expect("proxy provider");

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

        let provider = ProxyProvider::new(config(&server)).expect("proxy provider");
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

        let provider = ProxyProvider::new(config(&server)).expect("proxy provider");
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

        let provider = ProxyProvider::new(config(&server)).expect("proxy provider");
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

        let provider = ProxyProvider::new(config(&server)).expect("proxy provider");
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

        let provider = ProxyProvider::new(config(&server)).expect("proxy provider");
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
}
