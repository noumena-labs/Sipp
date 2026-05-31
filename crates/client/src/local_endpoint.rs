use cogentlm_engine::engine::{ChatRequest, CogentEngine};

use crate::dispatch::InferenceEndpoint;
use crate::{
    map, validate, CogentChatRequest, CogentEmbedRequest, CogentEmbeddingRun, CogentError,
    CogentQueryRequest, CogentTextRun, CogentTokenStream, EndpointCapabilities, EndpointRef,
};

pub(crate) struct LocalEndpoint {
    endpoint: EndpointRef,
    capabilities: EndpointCapabilities,
    engine: CogentEngine,
}

impl LocalEndpoint {
    pub(crate) fn new(
        endpoint: EndpointRef,
        capabilities: EndpointCapabilities,
        engine: CogentEngine,
    ) -> Self {
        Self {
            endpoint,
            capabilities,
            engine,
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

    fn query(&self, request: CogentQueryRequest) -> CogentTextRun {
        if let Err(error) = validate::local_query(&request) {
            return CogentTextRun::ready_err(error);
        }
        let endpoint = self.endpoint.clone();
        let run = match map::local_query_request(request) {
            Ok(request) => self.engine.query(request),
            Err(error) => return CogentTextRun::ready_err(error),
        };
        let (tokens, response) = run.into_parts();
        CogentTextRun::new(
            Box::pin(async move {
                response
                    .await
                    .map(|result| map::text_response(endpoint, result))
                    .map_err(CogentError::Local)
            }),
            CogentTokenStream::from_engine(tokens),
        )
    }

    fn chat(&self, request: CogentChatRequest) -> CogentTextRun {
        if let Err(error) = validate::local_chat(&request) {
            return CogentTextRun::ready_err(error);
        }
        let endpoint = self.endpoint.clone();
        let options = match map::local_chat_options(request.options, request.local) {
            Ok(options) => options,
            Err(error) => return CogentTextRun::ready_err(error),
        };
        let run = self.engine.chat(
            ChatRequest::new(request.messages)
                .options(options)
                .stream_tokens(request.stream_tokens),
        );
        let (tokens, response) = run.into_parts();
        CogentTextRun::new(
            Box::pin(async move {
                response
                    .await
                    .map(|result| map::text_response(endpoint, result))
                    .map_err(CogentError::Local)
            }),
            CogentTokenStream::from_engine(tokens),
        )
    }

    fn embed(&self, request: CogentEmbedRequest) -> CogentEmbeddingRun {
        if let Err(error) = validate::local_embed(&request) {
            return CogentEmbeddingRun::ready_err(error);
        }
        let endpoint = self.endpoint.clone();
        let run = self
            .engine
            .embed(map::local_embed_request(request.input, request.local))
            .into_response();
        CogentEmbeddingRun::new(Box::pin(async move {
            run.await
                .map(|result| map::embedding_response(endpoint, result))
                .map_err(CogentError::Local)
        }))
    }
}
