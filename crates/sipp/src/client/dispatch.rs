use crate::client::{
    EndpointCapabilities, EndpointRef, SippChatRequest, SippEmbedRequest, SippEmbeddingRun,
    SippQueryRequest, SippRequestContext, SippTextRun,
};

/// Typed inference endpoint registered with [`SippClient`](crate::client::SippClient).
pub trait InferenceEndpoint: Send + Sync {
    fn endpoint(&self) -> &EndpointRef;
    fn capabilities(&self) -> &EndpointCapabilities;

    fn query_with_context(
        &self,
        context: SippRequestContext,
        request: SippQueryRequest,
    ) -> SippTextRun;

    fn chat_with_context(
        &self,
        context: SippRequestContext,
        request: SippChatRequest,
    ) -> SippTextRun;

    fn embed_with_context(
        &self,
        context: SippRequestContext,
        request: SippEmbedRequest,
    ) -> SippEmbeddingRun;
}
