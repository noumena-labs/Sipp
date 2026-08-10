use std::sync::Arc;

use crate::engine::{
    ChatRequest, EmbedRequest, EngineAudioRun, EngineEmbeddingResponseFuture,
    EngineTextResponseFuture, EngineTokenBatches, ListenRequest, QueryRequest, SippEngine,
    SpeakRequest,
};

use crate::client::dispatch::{EndpointCloseFuture, InferenceEndpoint};
use crate::client::{
    map, validate, EndpointCapabilities, EndpointRef, SippAudioRun, SippChatRequest,
    SippEmbedRequest, SippEmbeddingRun, SippError, SippListenRequest, SippQueryRequest,
    SippRequestContext, SippSpeakRequest, SippTextRun, SippTokenBatches,
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
    fn close(&self) -> EndpointCloseFuture<'_>;
    fn query(&self, request: QueryRequest) -> LocalTextRun;
    fn listen(&self, request: ListenRequest) -> LocalTextRun;
    fn speak(&self, request: SpeakRequest) -> EngineAudioRun;
    fn chat(&self, request: ChatRequest) -> LocalTextRun;
    fn embed(&self, request: EmbedRequest) -> EngineEmbeddingResponseFuture;
}

impl LocalRuntime for SippEngine {
    fn close(&self) -> EndpointCloseFuture<'_> {
        Box::pin(async move { SippEngine::close(self).await.map_err(SippError::Local) })
    }

    fn query(&self, request: QueryRequest) -> LocalTextRun {
        let (tokens, response) = SippEngine::query(self, request).into_parts();
        LocalTextRun { tokens, response }
    }

    fn listen(&self, request: ListenRequest) -> LocalTextRun {
        let (tokens, response) = SippEngine::listen(self, request).into_parts();
        LocalTextRun { tokens, response }
    }

    fn speak(&self, request: SpeakRequest) -> EngineAudioRun {
        SippEngine::speak(self, request)
    }

    fn chat(&self, request: ChatRequest) -> LocalTextRun {
        let (tokens, response) = SippEngine::chat(self, request).into_parts();
        LocalTextRun { tokens, response }
    }

    fn embed(&self, request: EmbedRequest) -> EngineEmbeddingResponseFuture {
        SippEngine::embed(self, request).into_response()
    }
}

impl LocalEndpoint {
    pub(crate) fn new(
        endpoint: EndpointRef,
        capabilities: EndpointCapabilities,
        engine: SippEngine,
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

    fn text_run(&self, context: SippRequestContext, run: LocalTextRun) -> SippTextRun {
        let endpoint = self.endpoint.clone();
        SippTextRun::new(
            Box::pin(async move {
                run.response
                    .await
                    .map(|result| map::text_response(endpoint, context.request_id, result))
                    .map_err(SippError::Local)
            }),
            SippTokenBatches::from_engine(run.tokens),
        )
    }
}

impl InferenceEndpoint for LocalEndpoint {
    fn endpoint(&self) -> &EndpointRef {
        &self.endpoint
    }

    fn capabilities(&self) -> &EndpointCapabilities {
        &self.capabilities
    }

    fn close(&self) -> EndpointCloseFuture<'_> {
        self.runtime.close()
    }

    fn query_with_context(
        &self,
        context: SippRequestContext,
        request: SippQueryRequest,
    ) -> SippTextRun {
        if let Err(error) = validate::local_query(&request) {
            return SippTextRun::ready_err(error);
        }
        let run = match map::local_query_request(request) {
            Ok(request) => self.runtime.query(request),
            Err(error) => return SippTextRun::ready_err(error),
        };
        self.text_run(context, run)
    }

    fn listen_with_context(
        &self,
        context: SippRequestContext,
        request: SippListenRequest,
    ) -> SippTextRun {
        let request = match map::local_listen_request(request) {
            Ok(request) => request,
            Err(error) => return SippTextRun::ready_err(error),
        };
        let run = self.runtime.listen(request);
        self.text_run(context, run)
    }

    fn speak_with_context(
        &self,
        context: SippRequestContext,
        request: SippSpeakRequest,
    ) -> SippAudioRun {
        let endpoint = self.endpoint.clone();
        let run = self.runtime.speak(map::local_speak_request(request));
        let (response, cancellation) = run.into_parts();
        SippAudioRun::new_with_engine_cancellation(
            Box::pin(async move {
                response
                    .await
                    .map(|output| map::audio_response(endpoint, context.request_id, output))
                    .map_err(SippError::Local)
            }),
            cancellation,
        )
    }

    fn chat_with_context(
        &self,
        context: SippRequestContext,
        request: SippChatRequest,
    ) -> SippTextRun {
        if let Err(error) = validate::local_chat(&request) {
            return SippTextRun::ready_err(error);
        }
        let options = match map::local_chat_options(request.options, request.local) {
            Ok(options) => options,
            Err(error) => return SippTextRun::ready_err(error),
        };
        let run = self.runtime.chat(
            ChatRequest::new(request.messages)
                .options(options)
                .emit_tokens(request.emit_tokens),
        );
        self.text_run(context, run)
    }

    fn embed_with_context(
        &self,
        context: SippRequestContext,
        request: SippEmbedRequest,
    ) -> SippEmbeddingRun {
        if let Err(error) = validate::local_embed(&request) {
            return SippEmbeddingRun::ready_err(error);
        }
        let endpoint = self.endpoint.clone();
        let run = self
            .runtime
            .embed(map::local_embed_request(request.input, request.local));
        SippEmbeddingRun::new(Box::pin(async move {
            run.await
                .map(|result| map::embedding_response(endpoint, context.request_id, result))
                .map_err(SippError::Local)
        }))
    }
}
