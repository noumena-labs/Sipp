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
#[path = "../tests/providers/openai_tests.rs"]
mod openai_tests;
