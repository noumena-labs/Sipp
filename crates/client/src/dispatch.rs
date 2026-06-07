use crate::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentQueryRequest,
    CogentRequestContext, CogentTextRun, EndpointCapabilities, EndpointRef,
};

pub(crate) trait InferenceEndpoint: Send + Sync {
    fn endpoint(&self) -> &EndpointRef;
    fn capabilities(&self) -> &EndpointCapabilities;
    fn query(&self, request: CogentQueryRequest) -> CogentTextRun {
        self.query_with_context(CogentRequestContext::default(), request)
    }
    fn query_with_context(
        &self,
        _context: CogentRequestContext,
        request: CogentQueryRequest,
    ) -> CogentTextRun {
        self.query(request)
    }
    fn chat(&self, request: CogentChatRequest) -> CogentTextRun {
        self.chat_with_context(CogentRequestContext::default(), request)
    }
    fn chat_with_context(
        &self,
        _context: CogentRequestContext,
        request: CogentChatRequest,
    ) -> CogentTextRun {
        self.chat(request)
    }
    fn embed(&self, request: CogentEmbedRequest) -> CogentEmbeddingRun {
        self.embed_with_context(CogentRequestContext::default(), request)
    }
    fn embed_with_context(
        &self,
        _context: CogentRequestContext,
        request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun {
        self.embed(request)
    }
}
