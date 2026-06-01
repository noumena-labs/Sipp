use crate::{
    HttpTransport, OpenAiConfig, ProviderAuth, ProviderBackend, ProviderChatRequest,
    ProviderChatResponse, ProviderEmbedRequest, ProviderEmbeddingResponse, ProviderError,
    ProviderErrorKind, ProviderGenerateRequest, ProviderGenerateResponse, ProviderKind,
    ProviderModel, ProviderResponseMetadata, ProviderResult, ProviderStream, ProviderStreamEvent,
    ProviderTextOutput,
};
use async_trait::async_trait;

use super::common::{
    insert_finite_f32_option, insert_positive_u32_option, merge_provider_options, optional_u32,
    provider_body_error, provider_response_error, require_non_empty_field,
};
use super::openai_compat::{
    map_finish_reason, openai_chat_body, openai_chat_response_from_body, openai_embedding_body,
    openai_embedding_response_from_body, openai_model_from_value, openai_models_from_body,
    openai_stream_chat_body, openai_stream_events,
};

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const OPENAI_RESPONSES_TYPED_FIELDS: &[&str] = &[
    "model",
    "input",
    "max_output_tokens",
    "temperature",
    "top_p",
    "stop",
    "stream",
];

pub struct OpenAiProvider {
    transport: HttpTransport,
}

impl OpenAiProvider {
    pub fn new(config: OpenAiConfig) -> ProviderResult<Self> {
        let base_url = config
            .base_url
            .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_string());
        let transport = HttpTransport::new_with_options(
            ProviderKind::OpenAi,
            base_url,
            ProviderAuth::Bearer(config.api_key),
            Vec::new(),
            config.timeout,
        )?;
        Ok(Self { transport })
    }
}

#[async_trait]
impl ProviderBackend for OpenAiProvider {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAi
    }

    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>> {
        let response = self.transport.get_json("/models").await?;
        openai_models_from_body(&response.body, ProviderKind::OpenAi)
    }

    async fn get_model(&self, model: &str) -> ProviderResult<ProviderModel> {
        let response = self.transport.get_json(&format!("/models/{model}")).await?;
        openai_model_from_value(&response.body, ProviderKind::OpenAi)
    }

    async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse> {
        let body = openai_chat_body(&req, ProviderKind::OpenAi)?;
        let response = self.transport.post_json("/chat/completions", &body).await?;
        openai_chat_response_from_body(response.request_id, response.body, ProviderKind::OpenAi)
    }

    async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        let body = openai_responses_body(&req)?;
        let response = self.transport.post_json("/responses", &body).await?;
        openai_responses_response_from_body(response.request_id, response.body)
    }

    async fn embed(&self, req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse> {
        let body = openai_embedding_body(&req, ProviderKind::OpenAi)?;
        let response = self.transport.post_json("/embeddings", &body).await?;
        openai_embedding_response_from_body(
            response.request_id,
            response.body,
            ProviderKind::OpenAi,
        )
    }

    async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        let body = openai_stream_chat_body(&req, ProviderKind::OpenAi)?;
        let response = self
            .transport
            .post_json_stream("/chat/completions", &body)
            .await?;
        Ok(openai_stream_events(
            response.request_id,
            response.stream,
            ProviderKind::OpenAi,
        ))
    }
}

fn openai_responses_body(req: &ProviderGenerateRequest) -> ProviderResult<serde_json::Value> {
    require_non_empty_field(&req.model, "model", ProviderKind::OpenAi)?;
    if !req.options.stop.is_empty() {
        return Err(ProviderError::new(
            ProviderErrorKind::UnsupportedFeature,
            ProviderKind::OpenAi,
            "stop is not supported by the OpenAI Responses adapter",
        ));
    }

    let mut body = serde_json::Map::new();
    body.insert(
        "model".to_string(),
        serde_json::Value::String(req.model.clone()),
    );
    body.insert(
        "input".to_string(),
        serde_json::Value::String(req.prompt.clone()),
    );
    insert_positive_u32_option(
        &mut body,
        "max_output_tokens",
        req.options.max_tokens,
        ProviderKind::OpenAi,
    )?;
    insert_finite_f32_option(
        &mut body,
        "temperature",
        req.options.temperature,
        ProviderKind::OpenAi,
    )?;
    insert_finite_f32_option(&mut body, "top_p", req.options.top_p, ProviderKind::OpenAi)?;

    merge_provider_options(
        &mut body,
        &req.provider_options,
        OPENAI_RESPONSES_TYPED_FIELDS,
        ProviderKind::OpenAi,
    )?;
    Ok(serde_json::Value::Object(body))
}

fn openai_responses_response_from_body(
    request_id: Option<String>,
    body: serde_json::Value,
) -> ProviderResult<ProviderGenerateResponse> {
    if body.get("error").is_some_and(|value| !value.is_null()) {
        return Err(provider_body_error(
            body,
            ProviderKind::OpenAi,
            "OpenAI response error",
        ));
    }

    let text = response_output_text(&body)?;
    let finish_reason_raw = response_finish_reason_raw(&body);
    let finish_reason = map_finish_reason(finish_reason_raw.as_deref());
    let response_model = body
        .get("model")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| provider_response_error("response missing model", ProviderKind::OpenAi))?
        .to_string();
    let response_id = body
        .get("id")
        .and_then(serde_json::Value::as_str)
        .map(str::to_owned);
    let usage = body
        .get("usage")
        .filter(|value| !value.is_null())
        .map(openai_responses_usage)
        .transpose()?;

    Ok(ProviderGenerateResponse {
        result: ProviderTextOutput {
            text,
            finish_reason,
        },
        usage,
        metadata: ProviderResponseMetadata {
            provider: ProviderKind::OpenAi,
            model: response_model,
            request_id,
            response_id,
            finish_reason_raw,
            raw: body,
        },
    })
}

fn response_output_text(body: &serde_json::Value) -> ProviderResult<String> {
    let output = body
        .get("output")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| provider_response_error("response missing output", ProviderKind::OpenAi))?;
    let mut text = String::new();
    for item in output {
        let Some(content) = item.get("content").and_then(serde_json::Value::as_array) else {
            continue;
        };
        for part in content {
            if part.get("type").and_then(serde_json::Value::as_str) == Some("output_text") {
                let part_text = part
                    .get("text")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        provider_response_error("output_text missing text", ProviderKind::OpenAi)
                    })?;
                text.push_str(part_text);
            }
        }
    }

    // A tool-call-only Responses output carries no `output_text` part; surface
    // empty text and keep the tool items in `metadata.raw` rather than erroring.
    Ok(text)
}

fn response_finish_reason_raw(body: &serde_json::Value) -> Option<String> {
    body.pointer("/incomplete_details/reason")
        .and_then(serde_json::Value::as_str)
        .or_else(|| body.get("status").and_then(serde_json::Value::as_str))
        .map(str::to_owned)
}

fn openai_responses_usage(value: &serde_json::Value) -> ProviderResult<crate::TokenUsage> {
    Ok(crate::TokenUsage {
        input_tokens: optional_u32(value, "input_tokens", ProviderKind::OpenAi)?,
        output_tokens: optional_u32(value, "output_tokens", ProviderKind::OpenAi)?,
        total_tokens: optional_u32(value, "total_tokens", ProviderKind::OpenAi)?,
    })
}

#[cfg(test)]
mod tests {
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
}
