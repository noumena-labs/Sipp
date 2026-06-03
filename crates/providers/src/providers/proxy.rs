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

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "../tests/providers/proxy_tests.rs"]
mod proxy_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
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
