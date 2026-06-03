use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    AnthropicAdapter, AnthropicAdapterConfig, OpenAiAdapter, OpenAiAdapterConfig,
    OpenAiCompatibleAdapter, OpenAiCompatibleAdapterConfig, ProviderChatRequest,
    ProviderChatResponse, ProviderEmbedRequest, ProviderEmbeddingResponse, ProviderGenerateRequest,
    ProviderGenerateResponse, ProviderKind, ProviderModel, ProviderResult, ProviderStream,
    ProviderStreamEvent,
};

/// Server-side adapter contract used by the gateway provider package.
#[async_trait]
pub trait GatewayBackendAdapter: Send + Sync {
    /// Return the provider kind implemented by this adapter.
    fn kind(&self) -> ProviderKind;

    /// List upstream models exposed by the adapter.
    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>>;

    /// Fetch one upstream model by name.
    async fn get_model(&self, model: &str) -> ProviderResult<ProviderModel>;

    /// Run a chat-shaped generation request.
    async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse>;

    /// Run a raw prompt generation request.
    async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse>;

    /// Stream a raw prompt generation request.
    async fn stream_generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>>;

    /// Run an embedding request.
    async fn embed(&self, req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse>;

    /// Stream a chat-shaped generation request.
    async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>>;
}

/// Type-erased gateway adapter transport.
#[derive(Clone)]
pub struct GatewayAdapterTransport {
    backend: Arc<dyn GatewayBackendAdapter>,
}

impl GatewayAdapterTransport {
    /// Create a transport from an adapter implementation.
    pub fn from_backend(backend: Arc<dyn GatewayBackendAdapter>) -> Self {
        Self { backend }
    }

    /// Create an OpenAI-compatible adapter transport.
    pub fn openai_compatible(config: OpenAiCompatibleAdapterConfig) -> ProviderResult<Self> {
        Ok(Self::from_backend(Arc::new(OpenAiCompatibleAdapter::new(
            config,
        )?)))
    }

    /// Create an OpenAI adapter transport.
    pub fn openai(config: OpenAiAdapterConfig) -> ProviderResult<Self> {
        Ok(Self::from_backend(Arc::new(OpenAiAdapter::new(config)?)))
    }

    /// Create an Anthropic adapter transport.
    pub fn anthropic(config: AnthropicAdapterConfig) -> ProviderResult<Self> {
        Ok(Self::from_backend(Arc::new(AnthropicAdapter::new(config)?)))
    }

    /// Return the provider kind behind this transport.
    pub fn kind(&self) -> ProviderKind {
        self.backend.kind()
    }

    /// List upstream models exposed by the adapter.
    pub async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>> {
        self.backend.list_models().await
    }

    /// Fetch one upstream model by name.
    pub async fn get_model(&self, model: &str) -> ProviderResult<ProviderModel> {
        self.backend.get_model(model).await
    }

    /// Run a chat-shaped generation request.
    pub async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse> {
        self.backend.chat(req).await
    }

    /// Run a raw prompt generation request.
    pub async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        self.backend.generate(req).await
    }

    /// Stream a raw prompt generation request.
    pub async fn stream_generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.backend.stream_generate(req).await
    }

    /// Run an embedding request.
    pub async fn embed(
        &self,
        req: ProviderEmbedRequest,
    ) -> ProviderResult<ProviderEmbeddingResponse> {
        self.backend.embed(req).await
    }

    /// Stream a chat-shaped generation request.
    pub async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.backend.stream_chat(req).await
    }
}
