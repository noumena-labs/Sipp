use crate::{
    GatewayBackendAdapter, HttpTransport, OpenAiAdapterConfig, ProviderAuth, ProviderChatRequest,
    ProviderChatResponse, ProviderEmbedRequest, ProviderEmbeddingResponse, ProviderGenerateRequest,
    ProviderGenerateResponse, ProviderKind, ProviderModel, ProviderResult, ProviderStream,
    ProviderStreamEvent,
};
use async_trait::async_trait;

use super::openai_compat::{
    openai_chat_body, openai_chat_response_from_body, openai_completion_body,
    openai_completion_response_from_body, openai_completion_stream_events, openai_embedding_body,
    openai_embedding_response_from_body, openai_model_from_value, openai_models_from_body,
    openai_stream_chat_body, openai_stream_completion_body, openai_stream_events,
};

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

pub struct OpenAiAdapter {
    transport: HttpTransport,
}

impl OpenAiAdapter {
    pub fn new(config: OpenAiAdapterConfig) -> ProviderResult<Self> {
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
impl GatewayBackendAdapter for OpenAiAdapter {
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
        let body = openai_completion_body(&req, ProviderKind::OpenAi)?;
        let response = self.transport.post_json("/completions", &body).await?;
        openai_completion_response_from_body(
            response.request_id,
            response.body,
            ProviderKind::OpenAi,
        )
    }

    async fn stream_generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        let body = openai_stream_completion_body(&req, ProviderKind::OpenAi)?;
        let response = self
            .transport
            .post_json_stream("/completions", &body)
            .await?;
        Ok(openai_completion_stream_events(
            response.request_id,
            response.stream,
            ProviderKind::OpenAi,
        ))
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

#[cfg(test)]
#[path = "../tests/providers/openai_tests.rs"]
mod openai_tests;
