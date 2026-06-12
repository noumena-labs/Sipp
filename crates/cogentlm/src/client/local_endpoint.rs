use std::sync::Arc;

use crate::engine::{
    ChatRequest, CogentEngine, EmbedRequest, EngineEmbeddingResponseFuture,
    EngineTextResponseFuture, EngineTokenBatches, QueryRequest,
};

use crate::client::dispatch::InferenceEndpoint;
use crate::client::{
    map, validate, CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentError,
    CogentQueryRequest, CogentRequestContext, CogentTextRun, CogentTokenBatches,
    EndpointCapabilities, EndpointRef,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/client/local_endpoint_tests.rs"]
mod local_endpoint_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) struct LocalEndpoint {
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
    runtime: Arc<dyn LocalRuntime>,
}

struct LocalTextRun {
    tokens: Option<EngineTokenBatches>,
    response: EngineTextResponseFuture,
}

trait LocalRuntime: Send + Sync {
    fn query(&self, request: QueryRequest) -> LocalTextRun;
    fn chat(&self, request: ChatRequest) -> LocalTextRun;
    fn embed(&self, request: EmbedRequest) -> EngineEmbeddingResponseFuture;
}

impl LocalRuntime for CogentEngine {
    fn query(&self, request: QueryRequest) -> LocalTextRun {
        let (tokens, response) = CogentEngine::query(self, request).into_parts();
        LocalTextRun { tokens, response }
    }

    fn chat(&self, request: ChatRequest) -> LocalTextRun {
        let (tokens, response) = CogentEngine::chat(self, request).into_parts();
        LocalTextRun { tokens, response }
    }

    fn embed(&self, request: EmbedRequest) -> EngineEmbeddingResponseFuture {
        CogentEngine::embed(self, request).into_response()
    }
}

impl LocalEndpoint {
    pub(crate) fn new(
        endpoint: EndpointRef,
        capabilities: EndpointCapabilities,
        engine: CogentEngine,
    ) -> Self {
        Self::from_runtime(endpoint, capabilities, Arc::new(engine))
    }

    fn from_runtime(
        endpoint: EndpointRef,
        capabilities: EndpointCapabilities,
        runtime: Arc<dyn LocalRuntime>,
    ) -> Self {
        Self {
            endpoint,
            capabilities,
            runtime,
        }
    }
}

impl InferenceEndpoint for LocalEndpoint {
    fn endpoint(&self) -> &EndpointRef {
        &self.endpoint
    }

    fn capabilities(&self) -> &EndpointCapabilities {
        &self.capabilities
    }

    fn query_with_context(
        &self,
        context: CogentRequestContext,
        request: CogentQueryRequest,
    ) -> CogentTextRun {
        if let Err(error) = validate::local_query(&request) {
            return CogentTextRun::ready_err(error);
        }
        let endpoint = self.endpoint.clone();
        let run = match map::local_query_request(request) {
            Ok(request) => self.runtime.query(request),
            Err(error) => return CogentTextRun::ready_err(error),
        };
        CogentTextRun::new(
            Box::pin(async move {
                run.response
                    .await
                    .map(|result| map::text_response(endpoint, context.request_id, result))
                    .map_err(CogentError::Local)
            }),
            CogentTokenBatches::from_engine(run.tokens),
        )
    }

    fn chat_with_context(
        &self,
        context: CogentRequestContext,
        request: CogentChatRequest,
    ) -> CogentTextRun {
        if let Err(error) = validate::local_chat(&request) {
            return CogentTextRun::ready_err(error);
        }
        let endpoint = self.endpoint.clone();
        let options = match map::local_chat_options(request.options, request.local) {
            Ok(options) => options,
            Err(error) => return CogentTextRun::ready_err(error),
        };
        let run = self.runtime.chat(
            ChatRequest::new(request.messages)
                .options(options)
                .emit_tokens(request.emit_tokens),
        );
        CogentTextRun::new(
            Box::pin(async move {
                run.response
                    .await
                    .map(|result| map::text_response(endpoint, context.request_id, result))
                    .map_err(CogentError::Local)
            }),
            CogentTokenBatches::from_engine(run.tokens),
        )
    }

    fn embed_with_context(
        &self,
        context: CogentRequestContext,
        request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun {
        if let Err(error) = validate::local_embed(&request) {
            return CogentEmbeddingRun::ready_err(error);
        }
        let endpoint = self.endpoint.clone();
        let run = self
            .runtime
            .embed(map::local_embed_request(request.input, request.local));
        CogentEmbeddingRun::new(Box::pin(async move {
            run.await
                .map(|result| map::embedding_response(endpoint, context.request_id, result))
                .map_err(CogentError::Local)
        }))
    }
}
