use crate::{
    CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentQueryRequest, CogentTextRun,
    EndpointCapabilities, EndpointRef,
};

pub(crate) trait InferenceEndpoint: Send + Sync {
    fn endpoint(&self) -> &EndpointRef;
    fn capabilities(&self) -> &EndpointCapabilities;
    fn query(&self, request: CogentQueryRequest) -> CogentTextRun;
    fn chat(&self, request: CogentChatRequest) -> CogentTextRun;
    fn embed(&self, request: CogentEmbedRequest) -> CogentEmbeddingRun;
}
