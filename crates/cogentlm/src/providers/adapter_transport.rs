use std::sync::Arc;

use async_trait::async_trait;

use crate::providers::{
    AnthropicAdapter, AnthropicAdapterConfig, OpenAiAdapter, OpenAiAdapterConfig,
    OpenAiCompatibleAdapter, OpenAiCompatibleAdapterConfig, ProviderChatRequest,
    ProviderChatResponse, ProviderEmbedRequest, ProviderEmbeddingResponse, ProviderGenerateRequest,
    ProviderGenerateResponse, ProviderKind, ProviderModel, ProviderRequestContext, ProviderResult,
    ProviderStream, ProviderStreamEvent,
};

/// Server-side adapter contract used by the provider package.
#[async_trait]
pub trait ProviderBackend: Send + Sync {
    /// Return the provider kind implemented by this adapter.
    fn kind(&self) -> ProviderKind;

    /// List upstream models exposed by the adapter.
    async fn list_models(&self) -> ProviderResult<Vec<ProviderModel>>;

    /// Fetch one upstream model by name.
    async fn get_model(&self, model: &str) -> ProviderResult<ProviderModel>;

    /// Run a chat-shaped generation request.
    async fn chat(&self, req: ProviderChatRequest) -> ProviderResult<ProviderChatResponse>;

    /// Run chat with request-scoped correlation metadata.
    async fn chat_with_context(
        &self,
        _context: ProviderRequestContext,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderChatResponse> {
        self.chat(req).await
    }

    /// Run a raw prompt generation request.
    async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse>;

    /// Run generation with request-scoped correlation metadata.
    async fn generate_with_context(
        &self,
        _context: ProviderRequestContext,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        self.generate(req).await
    }

    /// Stream a raw prompt generation request.
    async fn stream_generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>>;

    /// Stream generation with request-scoped correlation metadata.
    async fn stream_generate_with_context(
        &self,
        _context: ProviderRequestContext,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.stream_generate(req).await
    }

    /// Run an embedding request.
    async fn embed(&self, req: ProviderEmbedRequest) -> ProviderResult<ProviderEmbeddingResponse>;

    /// Run embedding with request-scoped correlation metadata.
    async fn embed_with_context(
        &self,
        _context: ProviderRequestContext,
        req: ProviderEmbedRequest,
    ) -> ProviderResult<ProviderEmbeddingResponse> {
        self.embed(req).await
    }

    /// Stream a chat-shaped generation request.
    async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>>;

    /// Stream chat with request-scoped correlation metadata.
    async fn stream_chat_with_context(
        &self,
        _context: ProviderRequestContext,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.stream_chat(req).await
    }
}

/// Type-erased provider transport.
#[derive(Clone)]
pub struct ProviderTransport {
    backend: Arc<dyn ProviderBackend>,
}

impl ProviderTransport {
    /// Create a transport from an adapter implementation.
    pub fn from_backend(backend: Arc<dyn ProviderBackend>) -> Self {
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

    /// Run chat with request-scoped correlation metadata.
    pub async fn chat_with_context(
        &self,
        context: ProviderRequestContext,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderChatResponse> {
        self.backend.chat_with_context(context, req).await
    }

    /// Run a raw prompt generation request.
    pub async fn generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        self.backend.generate(req).await
    }

    /// Run generation with request-scoped correlation metadata.
    pub async fn generate_with_context(
        &self,
        context: ProviderRequestContext,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderGenerateResponse> {
        self.backend.generate_with_context(context, req).await
    }

    /// Stream a raw prompt generation request.
    pub async fn stream_generate(
        &self,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.backend.stream_generate(req).await
    }

    /// Stream generation with request-scoped correlation metadata.
    pub async fn stream_generate_with_context(
        &self,
        context: ProviderRequestContext,
        req: ProviderGenerateRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.backend
            .stream_generate_with_context(context, req)
            .await
    }

    /// Run an embedding request.
    pub async fn embed(
        &self,
        req: ProviderEmbedRequest,
    ) -> ProviderResult<ProviderEmbeddingResponse> {
        self.backend.embed(req).await
    }

    /// Run embedding with request-scoped correlation metadata.
    pub async fn embed_with_context(
        &self,
        context: ProviderRequestContext,
        req: ProviderEmbedRequest,
    ) -> ProviderResult<ProviderEmbeddingResponse> {
        self.backend.embed_with_context(context, req).await
    }

    /// Stream a chat-shaped generation request.
    pub async fn stream_chat(
        &self,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.backend.stream_chat(req).await
    }

    /// Stream chat with request-scoped correlation metadata.
    pub async fn stream_chat_with_context(
        &self,
        context: ProviderRequestContext,
        req: ProviderChatRequest,
    ) -> ProviderResult<ProviderStream<ProviderStreamEvent>> {
        self.backend.stream_chat_with_context(context, req).await
    }
}
