use crate::client::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentQueryRequest,
    CogentRequestContext, CogentTextRun, EndpointCapabilities, EndpointRef,
};

/// Typed inference endpoint registered with [`CogentClient`](crate::client::CogentClient).
pub trait InferenceEndpoint: Send + Sync {
    fn endpoint(&self) -> &EndpointRef;
    fn capabilities(&self) -> &EndpointCapabilities;

    fn query_with_context(
        &self,
        context: CogentRequestContext,
        request: CogentQueryRequest,
    ) -> CogentTextRun;

    fn chat_with_context(
        &self,
        context: CogentRequestContext,
        request: CogentChatRequest,
    ) -> CogentTextRun;

    fn embed_with_context(
        &self,
        context: CogentRequestContext,
        request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun;
}
