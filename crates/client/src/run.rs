use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use cogentlm_core::TokenBatch;
use cogentlm_engine::engine::EngineTokenStream;
#[cfg(feature = "providers")]
use futures_channel::mpsc;
use futures_core::Stream;

use crate::{CogentEmbeddingResponse, CogentError, CogentResult, CogentTextResponse};

#[cfg(feature = "providers")]
pub(crate) const TOKEN_STREAM_CHANNEL_CAPACITY: usize = 256;

/// Final text response future.
pub type CogentTextResponseFuture =
    Pin<Box<dyn Future<Output = CogentResult<CogentTextResponse>> + Send>>;

/// Final embedding response future.
pub type CogentEmbeddingResponseFuture =
    Pin<Box<dyn Future<Output = CogentResult<CogentEmbeddingResponse>> + Send>>;

/// Awaitable text run plus token stream owner.
pub struct CogentTextRun {
    response: CogentTextResponseFuture,
    tokens: CogentTokenStream,
}

impl CogentTextRun {
    pub(crate) fn new(response: CogentTextResponseFuture, tokens: CogentTokenStream) -> Self {
        Self { response, tokens }
    }

    pub(crate) fn ready_err(error: CogentError) -> Self {
        Self::new(
            Box::pin(async move { Err(error) }),
            CogentTokenStream::closed(),
        )
    }

    /// Borrow the token stream owned by this text run.
    pub fn tokens(&mut self) -> &mut CogentTokenStream {
        &mut self.tokens
    }

    /// Split the token stream from the final-response future.
    pub fn into_parts(self) -> (CogentTokenStream, CogentTextResponseFuture) {
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

/// Best-effort token stream for a text run.
pub struct CogentTokenStream {
    inner: TokenStreamInner,
}

enum TokenStreamInner {
    Empty,
    Local(EngineTokenStream),
    #[cfg(feature = "providers")]
    Receiver(mpsc::Receiver<TokenBatch>),
}

impl CogentTokenStream {
    pub(crate) fn closed() -> Self {
        Self {
            inner: TokenStreamInner::Empty,
        }
    }

    pub(crate) fn from_engine(stream: Option<EngineTokenStream>) -> Self {
        match stream {
            Some(stream) => Self {
                inner: TokenStreamInner::Local(stream),
            },
            None => Self::closed(),
        }
    }

    #[cfg(feature = "providers")]
    pub(crate) fn from_receiver(receiver: mpsc::Receiver<TokenBatch>) -> Self {
        Self {
            inner: TokenStreamInner::Receiver(receiver),
        }
    }
}

impl Stream for CogentTokenStream {
    type Item = TokenBatch;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match &mut self.inner {
            TokenStreamInner::Empty => Poll::Ready(None),
            TokenStreamInner::Local(stream) => Pin::new(stream).poll_next(cx),
            #[cfg(feature = "providers")]
            TokenStreamInner::Receiver(receiver) => Pin::new(receiver).poll_next(cx),
        }
    }
}
