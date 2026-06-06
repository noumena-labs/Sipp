use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use cogentlm_core::TokenBatch;
use cogentlm_engine::engine::EngineTokenBatches;
#[cfg(any(feature = "remote", feature = "providers"))]
use futures_channel::mpsc;
use futures_core::Stream;

use crate::{CogentEmbeddingResponse, CogentError, CogentResult, CogentTextResponse};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/run_tests.rs"]
mod run_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Final text response future.
pub type CogentTextResponseFuture =
    Pin<Box<dyn Future<Output = CogentResult<CogentTextResponse>> + Send>>;

/// Final embedding response future.
pub type CogentEmbeddingResponseFuture =
    Pin<Box<dyn Future<Output = CogentResult<CogentEmbeddingResponse>> + Send>>;

/// Awaitable text run plus token batches owner.
pub struct CogentTextRun {
    response: CogentTextResponseFuture,
    tokens: CogentTokenBatches,
}

impl CogentTextRun {
    pub(crate) fn new(response: CogentTextResponseFuture, tokens: CogentTokenBatches) -> Self {
        Self { response, tokens }
    }

    pub(crate) fn ready_err(error: CogentError) -> Self {
        Self::new(
            Box::pin(async move { Err(error) }),
            CogentTokenBatches::closed(),
        )
    }

    /// Borrow the token batches owned by this text run.
    pub fn tokens(&mut self) -> &mut CogentTokenBatches {
        &mut self.tokens
    }

    /// Split the token batches from the final-response future.
    pub fn into_parts(self) -> (CogentTokenBatches, CogentTextResponseFuture) {
        (self.tokens, self.response)
    }
}

impl Future for CogentTextRun {
    type Output = CogentResult<CogentTextResponse>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.response.as_mut().poll(cx)
    }
}

/// Awaitable embedding run.
pub struct CogentEmbeddingRun {
    response: CogentEmbeddingResponseFuture,
}

impl CogentEmbeddingRun {
    pub(crate) fn new(response: CogentEmbeddingResponseFuture) -> Self {
        Self { response }
    }

    pub(crate) fn ready_err(error: CogentError) -> Self {
        Self::new(Box::pin(async move { Err(error) }))
    }

    /// Convert the run into its final-response future.
    pub fn into_response(self) -> CogentEmbeddingResponseFuture {
        self.response
    }
}

impl Future for CogentEmbeddingRun {
    type Output = CogentResult<CogentEmbeddingResponse>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.response.as_mut().poll(cx)
    }
}

/// Exact token batches for a text run.
pub struct CogentTokenBatches {
    inner: TokenBatchSource,
}

enum TokenBatchSource {
    Empty,
    Local(EngineTokenBatches),
    #[cfg(any(feature = "remote", feature = "providers"))]
    Receiver(mpsc::UnboundedReceiver<TokenBatch>),
}

impl CogentTokenBatches {
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

    #[cfg(any(feature = "remote", feature = "providers"))]
    pub(crate) fn from_receiver(receiver: mpsc::UnboundedReceiver<TokenBatch>) -> Self {
        Self {
            inner: TokenBatchSource::Receiver(receiver),
        }
    }
}

impl Stream for CogentTokenBatches {
    type Item = TokenBatch;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match &mut self.inner {
            TokenBatchSource::Empty => Poll::Ready(None),
            TokenBatchSource::Local(stream) => Pin::new(stream).poll_next(cx),
            #[cfg(any(feature = "remote", feature = "providers"))]
            TokenBatchSource::Receiver(receiver) => Pin::new(receiver).poll_next(cx),
        }
    }
}
