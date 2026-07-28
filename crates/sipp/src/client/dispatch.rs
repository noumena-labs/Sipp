use std::future::Future;
use std::pin::Pin;

use crate::client::{
    EndpointCapabilities, EndpointRef, SippChatRequest, SippEmbedRequest, SippEmbeddingRun,
    SippQueryRequest, SippRequestContext, SippResult, SippTextRun,
};

pub(crate) type EndpointCloseFuture<'a> = Pin<Box<dyn Future<Output = SippResult<()>> + Send + 'a>>;

/// Typed inference endpoint registered with [`SippClient`](crate::client::SippClient).
pub trait InferenceEndpoint: Send + Sync {
    fn endpoint(&self) -> &EndpointRef;
    fn capabilities(&self) -> &EndpointCapabilities;

    fn close(&self) -> EndpointCloseFuture<'_> {
        Box::pin(async { Ok(()) })
    }

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
