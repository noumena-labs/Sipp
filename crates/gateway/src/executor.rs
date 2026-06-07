use std::sync::Arc;

use cogentlm_client::{
    CogentChatRequest, CogentClient, CogentEmbedRequest, CogentEmbeddingRun, CogentQueryRequest,
    CogentTextRun,
};

use crate::GatewayRequestContext;

/// Framework-neutral execution contract used by `GatewayAdapter`.
pub trait GatewayExecutor: Send + Sync {
    /// Execute a query request.
    fn query(&self, context: &GatewayRequestContext, request: CogentQueryRequest) -> CogentTextRun;

    /// Execute a chat request.
    fn chat(&self, context: &GatewayRequestContext, request: CogentChatRequest) -> CogentTextRun;

    /// Execute an embedding request.
    fn embed(
        &self,
        context: &GatewayRequestContext,
        request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun;
}

/// Gateway executor backed by the unified CogentLM client.
#[derive(Clone)]
pub struct CogentClientExecutor {
    client: Arc<CogentClient>,
}

impl CogentClientExecutor {
    /// Wrap a fully configured client.
    pub fn new(client: CogentClient) -> Self {
        Self {
            client: Arc::new(client),
        }
    }

    /// Wrap a shared client.
    pub fn from_shared(client: Arc<CogentClient>) -> Self {
        Self { client }
    }
}

impl GatewayExecutor for CogentClientExecutor {
    fn query(&self, context: &GatewayRequestContext, request: CogentQueryRequest) -> CogentTextRun {
        self.client
            .query_with_context(context.client_context(), request)
    }

    fn chat(&self, context: &GatewayRequestContext, request: CogentChatRequest) -> CogentTextRun {
        self.client
            .chat_with_context(context.client_context(), request)
    }

    fn embed(
        &self,
        context: &GatewayRequestContext,
        request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun {
        self.client
            .embed_with_context(context.client_context(), request)
    }
}
