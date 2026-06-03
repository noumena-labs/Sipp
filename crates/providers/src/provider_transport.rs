use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    AnthropicConfig, AnthropicProvider, OpenAiConfig, OpenAiProvider, ProviderChatRequest,
    ProviderChatResponse, ProviderEmbedRequest, ProviderEmbeddingResponse, ProviderGenerateRequest,
    ProviderGenerateResponse, ProviderKind, ProviderModel, ProviderResult, ProviderStream,
    ProviderStreamEvent, ProxyConfig, ProxyProvider,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "tests/provider_transport_tests.rs"]
mod provider_transport_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

#[async_trait]
pub trait ProviderBackend: Send + Sync {
    fn kind(&self) -> ProviderKind;

    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>>;

    async fn get_model(&self, model: &str) -> ProviderResult<ProviderModel>;

    async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse>;

    async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse>;

    async fn embed(&self, req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse>;

    async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>>;
}

#[derive(Clone)]
pub struct ProviderTransport {
    backend: Arc<dyn ProviderBackend>,
}

impl ProviderTransport {
    pub fn from_backend(backend: Arc<dyn ProviderBackend>) -> Self {
        Self { backend }
    }

    pub fn proxy(config: ProxyConfig) -> ProviderResult<Self> {
        Ok(Self::from_backend(Arc::new(ProxyProvider::new(config)?)))
    }

    pub fn openai(config: OpenAiConfig) -> ProviderResult<Self> {
        Ok(Self::from_backend(Arc::new(OpenAiProvider::new(config)?)))
    }

    pub fn anthropic(config: AnthropicConfig) -> ProviderResult<Self> {
        Ok(Self::from_backend(Arc::new(AnthropicProvider::new(
            config,
        )?)))
    }

    pub fn kind(&self) -> ProviderKind {
        self.backend.kind()
    }

    pub async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>> {
        self.backend.list_models().await
    }

    pub async fn get_model(&self, model: &str) -> ProviderResult<ProviderModel> {
        self.backend.get_model(model).await
    }

    pub async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse> {
        self.backend.chat(req).await
    }

    pub async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        self.backend.generate(req).await
    }

    pub async fn embed(
        &self,
        req: ProviderEmbedRequest,
    ) -> ProviderResult<ProviderEmbeddingResponse> {
        self.backend.embed(req).await
    }

    pub async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.backend.stream_chat(req).await
    }
}
