use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;

use crate::client::{
    CogentChatRequest, CogentClient, CogentEmbedRequest, CogentEmbeddingResponse,
    CogentEmbeddingRun, CogentQueryRequest, CogentTextResponse, CogentTextRun, EndpointRef,
};
use crate::core::{FinishReason, TokenBatch, TokenUsage};
use futures_util::future::{select, Either};
use futures_util::{stream, Stream, StreamExt};

use crate::gateway_core::{GatewayError, GatewayRequestContext, GatewayResult};

/// Typed inference capability executed by the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    /// Raw-prompt generation.
    Query,
    /// Message-shaped generation.
    Chat,
    /// Vector embedding.
    Embed,
}

/// Resolve an application target into a registered inference endpoint.
pub trait TargetResolver: Send + Sync {
    /// Resolve a target for one typed inference operation.
    fn resolve(
        &self,
        context: &GatewayRequestContext,
        target: &str,
        operation: Operation,
    ) -> GatewayResult<EndpointRef>;
}

/// Authorize a resolved endpoint execution.
pub trait Authorizer: Send + Sync {
    /// Authorize the request or return a policy error.
    fn authorize(
        &self,
        context: &GatewayRequestContext,
        target: &str,
        endpoint: &EndpointRef,
        operation: Operation,
    ) -> GatewayResult<()>;
}

/// Permit retained for the complete execution lifetime.
pub trait AdmissionPermit: Send {}

impl<T: Send> AdmissionPermit for T {}

/// Decide whether a resolved request may begin execution.
pub trait AdmissionController: Send + Sync {
    /// Acquire a permit retained until finite or streaming execution completes.
    fn acquire(
        &self,
        context: &GatewayRequestContext,
        target: &str,
        endpoint: &EndpointRef,
        operation: Operation,
    ) -> GatewayResult<Box<dyn AdmissionPermit>>;
}

/// Typed execution backend used by the pipeline.
pub trait GatewayExecutor: Send + Sync {
    /// Execute query inference.
    fn query(&self, context: &GatewayRequestContext, request: CogentQueryRequest) -> CogentTextRun;

    /// Execute chat inference.
    fn chat(&self, context: &GatewayRequestContext, request: CogentChatRequest) -> CogentTextRun;

    /// Execute embedding inference.
    fn embed(
        &self,
        context: &GatewayRequestContext,
        request: CogentEmbedRequest,
    ) -> CogentEmbeddingRun;
}

/// Executor backed by a configured [`CogentClient`].
#[derive(Clone)]
pub struct CogentClientExecutor {
    client: Arc<CogentClient>,
}

impl CogentClientExecutor {
    /// Wrap an owned client.
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

/// Authorizer that permits every resolved request.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllowAllAuthorizer;

impl Authorizer for AllowAllAuthorizer {
    fn authorize(
        &self,
        _context: &GatewayRequestContext,
        _target: &str,
        _endpoint: &EndpointRef,
        _operation: Operation,
    ) -> GatewayResult<()> {
        Ok(())
    }
}

/// Admission controller that never limits execution.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnlimitedAdmissionController;

impl AdmissionController for UnlimitedAdmissionController {
    fn acquire(
        &self,
        _context: &GatewayRequestContext,
        _target: &str,
        _endpoint: &EndpointRef,
        _operation: Operation,
    ) -> GatewayResult<Box<dyn AdmissionPermit>> {
        Ok(Box::new(()))
    }
}

/// Immutable protocol-neutral gateway execution pipeline.
#[derive(Clone)]
pub struct GatewayPipeline {
    resolver: Arc<dyn TargetResolver>,
    authorizer: Arc<dyn Authorizer>,
    admission: Arc<dyn AdmissionController>,
    executor: Arc<dyn GatewayExecutor>,
}

impl GatewayPipeline {
    /// Compose the pipeline from application-defined policy and execution.
    pub fn new(
        resolver: Arc<dyn TargetResolver>,
        authorizer: Arc<dyn Authorizer>,
        admission: Arc<dyn AdmissionController>,
        executor: Arc<dyn GatewayExecutor>,
    ) -> Self {
        Self {
            resolver,
            authorizer,
            admission,
            executor,
        }
    }

    /// Execute a finite query.
    pub async fn query(
        &self,
        context: &GatewayRequestContext,
        target: &str,
        mut request: CogentQueryRequest,
    ) -> GatewayResult<CogentTextResponse> {
        let (endpoint, permit) = self.prepare(context, target, Operation::Query)?;
        request.endpoint = Some(endpoint);
        let run = self.executor.query(context, request);
        context.cancellation.register(run.cancellation_handle());
        let result = run.await.map_err(GatewayError::from);
        drop(permit);
        result
    }

    /// Execute a streaming query.
    pub fn stream_query(
        &self,
        context: &GatewayRequestContext,
        target: &str,
        mut request: CogentQueryRequest,
    ) -> GatewayResult<GatewayStream> {
        let (endpoint, permit) = self.prepare(context, target, Operation::Query)?;
        request.endpoint = Some(endpoint);
        request.emit_tokens = true;
        Ok(text_stream(
            context,
            self.executor.query(context, request),
            permit,
        ))
    }

    /// Execute a finite chat request.
    pub async fn chat(
        &self,
        context: &GatewayRequestContext,
        target: &str,
        mut request: CogentChatRequest,
    ) -> GatewayResult<CogentTextResponse> {
        let (endpoint, permit) = self.prepare(context, target, Operation::Chat)?;
        request.endpoint = Some(endpoint);
        let run = self.executor.chat(context, request);
        context.cancellation.register(run.cancellation_handle());
        let result = run.await.map_err(GatewayError::from);
        drop(permit);
        result
    }

    /// Execute a streaming chat request.
    pub fn stream_chat(
        &self,
        context: &GatewayRequestContext,
        target: &str,
        mut request: CogentChatRequest,
    ) -> GatewayResult<GatewayStream> {
        let (endpoint, permit) = self.prepare(context, target, Operation::Chat)?;
        request.endpoint = Some(endpoint);
        request.emit_tokens = true;
        Ok(text_stream(
            context,
            self.executor.chat(context, request),
            permit,
        ))
    }

    /// Execute an embedding request.
    pub async fn embed(
        &self,
        context: &GatewayRequestContext,
        target: &str,
        mut request: CogentEmbedRequest,
    ) -> GatewayResult<CogentEmbeddingResponse> {
        let (endpoint, permit) = self.prepare(context, target, Operation::Embed)?;
        request.endpoint = Some(endpoint);
        let run = self.executor.embed(context, request);
        context.cancellation.register(run.cancellation_handle());
        let result = run.await.map_err(GatewayError::from);
        drop(permit);
        result
    }

    fn prepare(
        &self,
        context: &GatewayRequestContext,
        target: &str,
        operation: Operation,
    ) -> GatewayResult<(EndpointRef, Box<dyn AdmissionPermit>)> {
        let endpoint = self.resolver.resolve(context, target, operation)?;
        self.authorizer
            .authorize(context, target, &endpoint, operation)?;
        let permit = self
            .admission
            .acquire(context, target, &endpoint, operation)?;
        Ok((endpoint, permit))
    }
}

/// Stream returned by query and chat execution.
pub type GatewayStream = Pin<Box<dyn Stream<Item = GatewayResult<GatewayStreamEvent>> + Send>>;

/// Protocol-neutral text execution event.
#[derive(Debug, Clone, PartialEq)]
pub enum GatewayStreamEvent {
    /// Generated token batch.
    TokenBatch(TokenBatch),
    /// Final token usage.
    Usage(TokenUsage),
    /// Successful completion metadata.
    Finished {
        /// Completion reason.
        finish_reason: FinishReason,
        /// Final response metadata.
        metadata: crate::client::CogentResponseMetadata,
    },
}

struct TextStreamState {
    tokens: crate::client::CogentTokenBatches,
    response: Option<crate::client::CogentTextResponseFuture>,
    pending: VecDeque<GatewayResult<GatewayStreamEvent>>,
    terminal: bool,
    permit: Option<Box<dyn AdmissionPermit>>,
}

fn text_stream(
    context: &GatewayRequestContext,
    run: CogentTextRun,
    permit: Box<dyn AdmissionPermit>,
) -> GatewayStream {
    let (tokens, response, cancellation) = run.into_parts_with_cancel();
    context.cancellation.register(cancellation);
    let state = TextStreamState {
        tokens,
        response: Some(response),
        pending: VecDeque::new(),
        terminal: false,
        permit: Some(permit),
    };
    Box::pin(stream::unfold(state, |mut state| async move {
        if let Some(event) = state.pending.pop_front() {
            return Some((event, state));
        }
        if state.terminal {
            return None;
        }
        let response = state.response.take()?;
        match select(state.tokens.next(), response).await {
            Either::Left((Some(batch), response)) => {
                state.response = Some(response);
                Some((Ok(GatewayStreamEvent::TokenBatch(batch)), state))
            }
            Either::Left((None, response)) => {
                finish_stream(&mut state, response.await);
                state.pending.pop_front().map(|event| (event, state))
            }
            Either::Right((response, tokens)) => {
                drop(tokens);
                finish_stream(&mut state, response);
                state.pending.pop_front().map(|event| (event, state))
            }
        }
    }))
}

fn finish_stream(
    state: &mut TextStreamState,
    response: crate::client::CogentResult<CogentTextResponse>,
) {
    state.terminal = true;
    state.permit.take();
    match response {
        Ok(response) => {
            if let Some(usage) = response.usage {
                state
                    .pending
                    .push_back(Ok(GatewayStreamEvent::Usage(usage)));
            }
            state.pending.push_back(Ok(GatewayStreamEvent::Finished {
                finish_reason: response.finish_reason,
                metadata: response.metadata,
            }));
        }
        Err(error) => state.pending.push_back(Err(error.into())),
    }
}

#[cfg(test)]
#[path = "../tests/gateway_core/pipeline_tests.rs"]
mod pipeline_tests;
