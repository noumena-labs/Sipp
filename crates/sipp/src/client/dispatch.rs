use std::future::Future;
use std::pin::Pin;

use crate::client::{
    EndpointCapabilities, EndpointRef, SippAudioRun, SippChatRequest, SippEmbedRequest,
    SippEmbeddingRun, SippError, SippListenRequest, SippQueryRequest, SippRequestContext,
    SippResult, SippSpeakRequest, SippTextRun,
};
use crate::core::Operation;

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

    fn listen_with_context(
        &self,
        _context: SippRequestContext,
        _request: SippListenRequest,
    ) -> SippTextRun {
        SippTextRun::ready_err(unsupported(self.endpoint(), Operation::Listen))
    }

    fn speak_with_context(
        &self,
        _context: SippRequestContext,
        _request: SippSpeakRequest,
    ) -> SippAudioRun {
        SippAudioRun::ready_err(unsupported(self.endpoint(), Operation::Speak))
    }
}

pub(crate) fn unsupported(endpoint: &EndpointRef, operation: Operation) -> SippError {
    SippError::UnsupportedOperation {
        endpoint: endpoint.clone(),
        operation: operation.as_str(),
    }
}
