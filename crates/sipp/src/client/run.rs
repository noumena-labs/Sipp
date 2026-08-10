use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use crate::core::TokenBatch;
use crate::engine::{EngineCancellation, EngineTokenBatches};
use futures::future::{select, Either};
#[cfg(not(target_family = "wasm"))]
use futures_channel::mpsc;
use futures_channel::oneshot;
use futures_core::Stream;

use crate::client::{
    SippAudioResponse, SippEmbeddingResponse, SippError, SippResult, SippTextResponse,
};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/client/run_tests.rs"]
mod run_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Final text response future.
pub type SippTextResponseFuture =
    Pin<Box<dyn Future<Output = SippResult<SippTextResponse>> + Send>>;

/// Final embedding response future.
pub type SippEmbeddingResponseFuture =
    Pin<Box<dyn Future<Output = SippResult<SippEmbeddingResponse>> + Send>>;

/// Final synthesized-audio response future.
pub type SippAudioResponseFuture =
    Pin<Box<dyn Future<Output = SippResult<SippAudioResponse>> + Send>>;

/// Stable reason attached to explicit request cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SippCancellationReason {
    /// The application explicitly cancelled the request.
    CallerCancelled,
    /// The downstream HTTP client disconnected.
    ClientDisconnected,
    /// The hosting server is shutting down.
    ServerShutdown,
    /// The request exceeded an application deadline.
    DeadlineExceeded,
}

impl SippCancellationReason {
    /// Return the stable cancellation label.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CallerCancelled => "caller_cancelled",
            Self::ClientDisconnected => "client_disconnected",
            Self::ServerShutdown => "server_shutdown",
            Self::DeadlineExceeded => "deadline_exceeded",
        }
    }
}

/// Cloneable handle that cancels an in-flight client run.
#[derive(Clone)]
pub struct SippCancellationHandle {
    sender: Arc<Mutex<Option<oneshot::Sender<SippCancellationReason>>>>,
    engine: Option<EngineCancellation>,
}

impl SippCancellationHandle {
    /// Cancel the run if it has not already completed or been cancelled.
    pub fn cancel(&self, reason: SippCancellationReason) {
        if let Some(engine) = &self.engine {
            engine.cancel();
        }
        let Ok(mut sender) = self.sender.lock() else {
            return;
        };
        if let Some(sender) = sender.take() {
            let _ = sender.send(reason);
        }
    }
}

struct CancellableResponse<T> {
    response: Pin<Box<dyn Future<Output = SippResult<T>> + Send>>,
    cancellation: SippCancellationHandle,
}

impl<T> CancellableResponse<T>
where
    T: Send + 'static,
{
    fn new(
        response: Pin<Box<dyn Future<Output = SippResult<T>> + Send>>,
        engine: Option<EngineCancellation>,
    ) -> Self {
        let (sender, receiver) = oneshot::channel();
        let cancellation = SippCancellationHandle {
            sender: Arc::new(Mutex::new(Some(sender))),
            engine,
        };
        let response = Box::pin(async move {
            match select(receiver, response).await {
                Either::Left((Ok(reason), response)) => {
                    drop(response);
                    Err(SippError::Cancelled { reason })
                }
                Either::Left((Err(_), response)) => response.await,
                Either::Right((result, _)) => result,
            }
        });
        Self {
            response,
            cancellation,
        }
    }
}

impl<T> Future for CancellableResponse<T> {
    type Output = SippResult<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.response.as_mut().poll(cx)
    }
}

/// Awaitable text run plus token batches owner.
pub struct SippTextRun {
    response: CancellableResponse<SippTextResponse>,
    tokens: SippTokenBatches,
}

impl SippTextRun {
    pub(crate) fn new(response: SippTextResponseFuture, tokens: SippTokenBatches) -> Self {
        Self {
            response: CancellableResponse::new(response, None),
            tokens,
        }
    }

    /// Create a finite text run from a response future.
    pub fn from_response(response: SippTextResponseFuture) -> Self {
        Self::new(response, SippTokenBatches::closed())
    }

    /// Create a text run from token batches and a final response future.
    pub fn from_parts(tokens: SippTokenBatches, response: SippTextResponseFuture) -> Self {
        Self::new(response, tokens)
    }

    pub(crate) fn ready_err(error: SippError) -> Self {
        Self::new(
            Box::pin(async move { Err(error) }),
            SippTokenBatches::closed(),
        )
    }

    /// Borrow the token batches owned by this text run.
    pub fn tokens(&mut self) -> &mut SippTokenBatches {
        &mut self.tokens
    }

    /// Return a handle that can cancel this run from another task.
    pub fn cancellation_handle(&self) -> SippCancellationHandle {
        self.response.cancellation.clone()
    }

    /// Cancel this run.
    pub fn cancel(&self, reason: SippCancellationReason) {
        self.response.cancellation.cancel(reason);
    }

    /// Split the token batches from the final-response future.
    pub fn into_parts(self) -> (SippTokenBatches, SippTextResponseFuture) {
        (self.tokens, self.response.response)
    }

    /// Split the run while retaining an explicit cancellation handle.
    pub fn into_parts_with_cancel(
        self,
    ) -> (
        SippTokenBatches,
        SippTextResponseFuture,
        SippCancellationHandle,
    ) {
        (
            self.tokens,
            self.response.response,
            self.response.cancellation,
        )
    }
}

impl Future for SippTextRun {
    type Output = SippResult<SippTextResponse>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.response).poll(cx)
    }
}

/// Awaitable embedding run.
pub struct SippEmbeddingRun {
    response: CancellableResponse<SippEmbeddingResponse>,
}

impl SippEmbeddingRun {
    pub(crate) fn new(response: SippEmbeddingResponseFuture) -> Self {
        Self {
            response: CancellableResponse::new(response, None),
        }
    }

    /// Create an embedding run from a response future.
    pub fn from_response(response: SippEmbeddingResponseFuture) -> Self {
        Self::new(response)
    }

    pub(crate) fn ready_err(error: SippError) -> Self {
        Self::new(Box::pin(async move { Err(error) }))
    }

    /// Return a handle that can cancel this run from another task.
    pub fn cancellation_handle(&self) -> SippCancellationHandle {
        self.response.cancellation.clone()
    }

    /// Cancel this run.
    pub fn cancel(&self, reason: SippCancellationReason) {
        self.response.cancellation.cancel(reason);
    }

    /// Convert the run into its final-response future.
    pub fn into_response(self) -> SippEmbeddingResponseFuture {
        self.response.response
    }

    /// Split the response future from its cancellation handle.
    pub fn into_parts(self) -> (SippEmbeddingResponseFuture, SippCancellationHandle) {
        (self.response.response, self.response.cancellation)
    }
}

impl Future for SippEmbeddingRun {
    type Output = SippResult<SippEmbeddingResponse>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.response).poll(cx)
    }
}

/// Awaitable synthesized-audio run.
pub struct SippAudioRun {
    response: CancellableResponse<SippAudioResponse>,
}

impl SippAudioRun {
    fn new(response: SippAudioResponseFuture) -> Self {
        Self {
            response: CancellableResponse::new(response, None),
        }
    }

    pub(crate) fn new_with_engine_cancellation(
        response: SippAudioResponseFuture,
        cancellation: EngineCancellation,
    ) -> Self {
        Self {
            response: CancellableResponse::new(response, Some(cancellation)),
        }
    }

    /// Create an audio run from a response future.
    pub fn from_response(response: SippAudioResponseFuture) -> Self {
        Self::new(response)
    }

    pub(crate) fn ready_err(error: SippError) -> Self {
        Self::new(Box::pin(async move { Err(error) }))
    }

    /// Return a handle that can cancel this run from another task.
    pub fn cancellation_handle(&self) -> SippCancellationHandle {
        self.response.cancellation.clone()
    }

    /// Cancel this run.
    pub fn cancel(&self, reason: SippCancellationReason) {
        self.response.cancellation.cancel(reason);
    }

    /// Convert the run into its final-response future.
    pub fn into_response(self) -> SippAudioResponseFuture {
        self.response.response
    }

    /// Split the response future from its cancellation handle.
    pub fn into_parts(self) -> (SippAudioResponseFuture, SippCancellationHandle) {
        (self.response.response, self.response.cancellation)
    }
}

impl Future for SippAudioRun {
    type Output = SippResult<SippAudioResponse>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut self.response).poll(cx)
    }
}

/// Exact token batches for a text run.
pub struct SippTokenBatches {
    inner: TokenBatchSource,
}

enum TokenBatchSource {
    Empty,
    Local(EngineTokenBatches),
    #[cfg(not(target_family = "wasm"))]
    Receiver(mpsc::UnboundedReceiver<TokenBatch>),
    External(Pin<Box<dyn Stream<Item = TokenBatch> + Send>>),
}

impl SippTokenBatches {
    pub(crate) fn closed() -> Self {
        Self {
            inner: TokenBatchSource::Empty,
        }
    }

    pub(crate) fn from_engine(stream: Option<EngineTokenBatches>) -> Self {
        match stream {
            Some(stream) => Self {
                inner: TokenBatchSource::Local(stream),
            },
            None => Self::closed(),
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn from_receiver(receiver: mpsc::UnboundedReceiver<TokenBatch>) -> Self {
        Self {
            inner: TokenBatchSource::Receiver(receiver),
        }
    }

    /// Create token batches from an endpoint-owned stream.
    pub fn from_stream(stream: Pin<Box<dyn Stream<Item = TokenBatch> + Send>>) -> Self {
        Self {
            inner: TokenBatchSource::External(stream),
        }
    }
}

impl Stream for SippTokenBatches {
    type Item = TokenBatch;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match &mut self.inner {
            TokenBatchSource::Empty => Poll::Ready(None),
            TokenBatchSource::Local(stream) => Pin::new(stream).poll_next(cx),
            #[cfg(not(target_family = "wasm"))]
            TokenBatchSource::Receiver(receiver) => Pin::new(receiver).poll_next(cx),
            TokenBatchSource::External(stream) => stream.as_mut().poll_next(cx),
        }
    }
}
