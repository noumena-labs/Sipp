use async_trait::async_trait;

use super::openai_compat::{
    openai_chat_body, openai_chat_response_from_body, openai_completion_body,
    openai_completion_response_from_body, openai_completion_stream_events, openai_embedding_body,
    openai_embedding_response_from_body, openai_model_from_value, openai_models_from_body,
    openai_stream_chat_body, openai_stream_completion_body, openai_stream_events,
};
use crate::{
    HttpTransport, OpenAiCompatibleAdapterConfig, OpenAiCompatibleProtocol, ProviderBackend,
    ProviderChatRequest, ProviderChatResponse, ProviderEmbedRequest, ProviderEmbeddingResponse,
    ProviderGenerateRequest, ProviderGenerateResponse, ProviderKind, ProviderModel, ProviderResult,
    ProviderStream, ProviderStreamEvent,
};

pub struct OpenAiCompatibleAdapter {
    transport: HttpTransport,
}

impl OpenAiCompatibleAdapter {
    pub fn new(config: OpenAiCompatibleAdapterConfig) -> ProviderResult<Self> {
        let OpenAiCompatibleAdapterConfig {
            base_url,
            auth,
            protocol,
            static_headers,
            timeout,
        } = config;
        match protocol {
            OpenAiCompatibleProtocol::OpenAiCompatible => {}
        }

        let transport = HttpTransport::new_with_options(
            ProviderKind::OpenAiCompatible,
            base_url,
            auth,
            static_headers,
            timeout,
        )?;
        Ok(Self { transport })
    }
}

#[async_trait]
impl ProviderBackend for OpenAiCompatibleAdapter {
    fn kind(&self) -> ProviderKind {
        ProviderKind::OpenAiCompatible
    }

    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>> {
        let response = self.transport.get_json("/models").await?;
        openai_models_from_body(&response.body, ProviderKind::OpenAiCompatible)
    }

    async fn get_model(&self, model: &str) -> ProviderResult<ProviderModel> {
        let response = self.transport.get_json(&format!("/models/{model}")).await?;
        openai_model_from_value(&response.body, ProviderKind::OpenAiCompatible)
    }

    async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse> {
        let body = openai_chat_body(&req, ProviderKind::OpenAiCompatible)?;
        let response = self.transport.post_json("/chat/completions", &body).await?;
        openai_chat_response_from_body(
            response.request_id,
            response.body,
            ProviderKind::OpenAiCompatible,
        )
    }

    async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        let body = openai_completion_body(&req, ProviderKind::OpenAiCompatible)?;
        let response = self.transport.post_json("/completions", &body).await?;
        openai_completion_response_from_body(
            response.request_id,
            response.body,
            ProviderKind::OpenAiCompatible,
        )
    }

    async fn stream_generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        let body = openai_stream_completion_body(&req, ProviderKind::OpenAiCompatible)?;
        let response = self
            .transport
            .post_json_stream("/completions", &body)
            .await?;
        Ok(openai_completion_stream_events(
            response.request_id,
            response.stream,
            ProviderKind::OpenAiCompatible,
        ))
    }

    async fn embed(&self, req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse> {
        let body = openai_embedding_body(&req, ProviderKind::OpenAiCompatible)?;
        let response = self.transport.post_json("/embeddings", &body).await?;
        openai_embedding_response_from_body(
            response.request_id,
            response.body,
            ProviderKind::OpenAiCompatible,
        )
    }

    async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        let body = openai_stream_chat_body(&req, ProviderKind::OpenAiCompatible)?;
        let response = self
            .transport
            .post_json_stream("/chat/completions", &body)
            .await?;
        Ok(openai_stream_events(
            response.request_id,
            response.stream,
            ProviderKind::OpenAiCompatible,
        ))
    }
}

#[cfg(test)]
#[path = "../tests/providers/openai_compatible_adapter_tests.rs"]
mod openai_compatible_adapter_tests;
