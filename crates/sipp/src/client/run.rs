use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use crate::core::TokenBatch;
use crate::engine::EngineTokenBatches;
use futures::future::{select, Either};
use futures_channel::mpsc;
use futures_channel::oneshot;
use futures_core::Stream;

use crate::client::{SippEmbeddingResponse, SippError, SippResult, SippTextResponse};

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
}

impl SippCancellationHandle {
    /// Cancel the run if it has not already completed or been cancelled.
    pub fn cancel(&self, reason: SippCancellationReason) {
        let Ok(mut sender) = self.sender.lock() else {
            return;
        };
        if let Some(sender) = sender.take() {
            let _ = sender.send(reason);
        }
    }
}

/// Awaitable text run plus token batches owner.
pub struct SippTextRun {
    response: SippTextResponseFuture,
    tokens: SippTokenBatches,
    cancellation: SippCancellationHandle,
}

impl SippTextRun {
    pub(crate) fn new(response: SippTextResponseFuture, tokens: SippTokenBatches) -> Self {
        let (response, cancellation) = cancellable_response(response);
        Self {
            response,
            tokens,
            cancellation,
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
        self.cancellation.clone()
    }

    /// Cancel this run.
    pub fn cancel(&self, reason: SippCancellationReason) {
        self.cancellation.cancel(reason);
    }

    /// Split the token batches from the final-response future.
    pub fn into_parts(self) -> (SippTokenBatches, SippTextResponseFuture) {
        (self.tokens, self.response)
    }

    /// Split the run while retaining an explicit cancellation handle.
    pub fn into_parts_with_cancel(
        self,
    ) -> (
        SippTokenBatches,
        SippTextResponseFuture,
        SippCancellationHandle,
    ) {
        (self.tokens, self.response, self.cancellation)
    }
}

impl Future for SippTextRun {
    type Output = SippResult<SippTextResponse>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.response.as_mut().poll(cx)
    }
}

/// Awaitable embedding run.
pub struct SippEmbeddingRun {
    response: SippEmbeddingResponseFuture,
    cancellation: SippCancellationHandle,
}

impl SippEmbeddingRun {
    pub(crate) fn new(response: SippEmbeddingResponseFuture) -> Self {
        let (response, cancellation) = cancellable_response(response);
        Self {
            response,
            cancellation,
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
        self.cancellation.clone()
    }

    /// Cancel this run.
    pub fn cancel(&self, reason: SippCancellationReason) {
        self.cancellation.cancel(reason);
    }

    /// Convert the run into its final-response future.
    pub fn into_response(self) -> SippEmbeddingResponseFuture {
        self.response
    }

    /// Split the response future from its cancellation handle.
    pub fn into_parts(self) -> (SippEmbeddingResponseFuture, SippCancellationHandle) {
        (self.response, self.cancellation)
    }
}

impl Future for SippEmbeddingRun {
    type Output = SippResult<SippEmbeddingResponse>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.response.as_mut().poll(cx)
    }
}

/// Exact token batches for a text run.
pub struct SippTokenBatches {
    inner: TokenBatchSource,
}

enum TokenBatchSource {
    Empty,
    Local(EngineTokenBatches),
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
            TokenBatchSource::Receiver(receiver) => Pin::new(receiver).poll_next(cx),
            TokenBatchSource::External(stream) => stream.as_mut().poll_next(cx),
        }
    }
}

fn cancellable_response<T>(
    response: Pin<Box<dyn Future<Output = SippResult<T>> + Send>>,
) -> (
    Pin<Box<dyn Future<Output = SippResult<T>> + Send>>,
    SippCancellationHandle,
)
where
    T: Send + 'static,
{
    let (sender, receiver) = oneshot::channel();
    let cancellation = SippCancellationHandle {
        sender: Arc::new(Mutex::new(Some(sender))),
    };
    let response = Box::pin(async move {
        match select(response, receiver).await {
            Either::Left((result, _)) => result,
            Either::Right((Ok(reason), response)) => {
                drop(response);
                Err(SippError::Cancelled { reason })
            }
            Either::Right((Err(_), response)) => response.await,
        }
    });
    (response, cancellation)
}
