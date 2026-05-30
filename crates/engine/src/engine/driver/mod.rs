//! Engine driver thread: owns the InferenceRuntime, accepts commands over a channel, and emits events.

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::{mpsc, Arc, Mutex};
use std::task::{Context, Poll};
use std::thread::{self, JoinHandle};

use crate::defaults::{BYTES_PER_KIB, DEFAULT_MODEL_FILE_NAME};
use crate::engine::{
    protocol::{
        EmbedRequest, EmbeddingResult, EngineEvent, EngineState, GenerationResult, ModelState,
    },
    NativeRuntimeConfig,
};
use crate::error::{Error, Result};
use crate::runtime::InferenceRuntime;
use futures_channel::{mpsc as futures_mpsc, oneshot};
use futures_core::Stream;

mod events;
mod request;
mod stats;
mod thread_loop;
mod token_stream;

pub use request::{ChatMessage, ChatRequest, ChatRole, QueryOptions, QueryRequest};
use stats::{embedding_result_from_response, generation_result_from_response};
use thread_loop::{run_engine_thread, EngineThreadCommand};

pub type EngineEventReceiver = mpsc::Receiver<EngineEvent>;
type EngineEventSubscribers = Arc<Mutex<Vec<mpsc::Sender<EngineEvent>>>>;

const TOKEN_BATCH_MAX_FRAMES: usize = 64;
const TOKEN_BATCH_MAX_BYTES: usize = 64 * BYTES_PER_KIB;
const ENGINE_THREAD_CLOSED: &str = "engine thread is closed";
const ENGINE_THREAD_STOPPED_BEFORE_RESPONSE: &str = "engine thread stopped before responding";
const ENGINE_THREAD_NAME: &str = "cogentlm-engine";
const ENGINE_THREAD_SPAWN_FAILED: &str = "failed to spawn engine thread";
const ENGINE_THREAD_STOPPED_DURING_LOAD: &str = "engine thread stopped during load";
const ENGINE_THREAD_PANICKED: &str = "engine thread panicked";

/// Cloneable handle to the synchronous engine actor.
#[derive(Clone)]
pub struct CogentEngine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    command_tx: mpsc::Sender<EngineThreadCommand>,
    event_subscribers: EngineEventSubscribers,
    _driver: JoinHandle<()>,
}

/// Future returned by [`CogentEngine::load`].
///
/// Dropping this future after the driver thread starts requests shutdown and
/// hands the thread to a background reaper; native model loading is not
/// preempted.
pub struct EngineLoad {
    command_tx: Option<mpsc::Sender<EngineThreadCommand>>,
    init_rx: Option<oneshot::Receiver<Result<()>>>,
    event_subscribers: EngineEventSubscribers,
    join_handle: Option<JoinHandle<()>>,
}

/// Awaitable text generation run returned by [`CogentEngine::query`] and
/// [`CogentEngine::chat`].
pub struct EngineTextRun {
    response: EngineResponse<crate::runtime::request::GenerateResponse>,
    tokens: Option<EngineTokenStream>,
    _engine: Arc<EngineInner>,
}

/// Awaitable embedding run returned by [`CogentEngine::embed`].
pub struct EngineEmbeddingRun {
    response: EngineResponse<crate::runtime::request::GenerateResponse>,
    _engine: Arc<EngineInner>,
}

/// Best-effort stream of token batches owned by an [`EngineTextRun`].
pub struct EngineTokenStream {
    rx: futures_mpsc::Receiver<cogentlm_core::TokenBatch>,
}

/// Boxed final-response future returned when a text run is split into parts.
pub type EngineTextResponseFuture = Pin<Box<dyn Future<Output = Result<GenerationResult>> + Send>>;
/// Boxed final-response future returned when an embedding run is split.
pub type EngineEmbeddingResponseFuture =
    Pin<Box<dyn Future<Output = Result<EmbeddingResult>> + Send>>;

enum EngineResponse<T> {
    Pending(oneshot::Receiver<Result<T>>),
    Ready(Option<Result<T>>),
}

impl<T> Unpin for EngineResponse<T> {}

impl CogentEngine {
    /// Start loading a model on a driver thread and return an awaitable load handle.
    pub fn load(model_path: impl AsRef<Path>, config: NativeRuntimeConfig) -> EngineLoad {
        let model_path = model_path.as_ref().to_path_buf();
        EngineLoad::spawn(model_path, config)
    }

    /// Submit a raw prompt generation request and return its run handle.
    pub fn query(&self, request: QueryRequest) -> EngineTextRun {
        let (response_tx, response_rx) = oneshot::channel();
        let (token_tx, tokens) = token_channel(request.stream_tokens);
        let response = match self.inner.command_tx.send(EngineThreadCommand::Generate(
            request,
            response_tx,
            token_tx,
        )) {
            Ok(()) => EngineResponse::Pending(response_rx),
            Err(_) => EngineResponse::ready_err(runtime_command(ENGINE_THREAD_CLOSED)),
        };
        EngineTextRun {
            response,
            tokens,
            _engine: Arc::clone(&self.inner),
        }
    }

    /// Submit a chat generation request and return its run handle.
    pub fn chat(&self, request: ChatRequest) -> EngineTextRun {
        let (response_tx, response_rx) = oneshot::channel();
        let (token_tx, tokens) = token_channel(request.stream_tokens);
        let response = match self
            .inner
            .command_tx
            .send(EngineThreadCommand::GenerateChat(
                request,
                response_tx,
                token_tx,
            )) {
            Ok(()) => EngineResponse::Pending(response_rx),
            Err(_) => EngineResponse::ready_err(runtime_command(ENGINE_THREAD_CLOSED)),
        };
        EngineTextRun {
            response,
            tokens,
            _engine: Arc::clone(&self.inner),
        }
    }

    /// Submit an embedding request and return its run handle.
    pub fn embed(&self, request: EmbedRequest) -> EngineEmbeddingRun {
        let (response_tx, response_rx) = oneshot::channel();
        let response = match self
            .inner
            .command_tx
            .send(EngineThreadCommand::Embed(request, response_tx))
        {
            Ok(()) => EngineResponse::Pending(response_rx),
            Err(_) => EngineResponse::ready_err(runtime_command(ENGINE_THREAD_CLOSED)),
        };
        EngineEmbeddingRun {
            response,
            _engine: Arc::clone(&self.inner),
        }
    }

    /// Fetch a point-in-time state snapshot from the driver thread.
    pub async fn state(&self) -> Result<EngineState> {
        let (response_tx, response_rx) = oneshot::channel();
        if self
            .inner
            .command_tx
            .send(EngineThreadCommand::GetState(response_tx))
            .is_err()
        {
            return Err(runtime_command(ENGINE_THREAD_CLOSED));
        }
        response_rx
            .await
            .map_err(|_| runtime_command(ENGINE_THREAD_STOPPED_BEFORE_RESPONSE))?
    }

    /// Close the engine actor and wait until the runtime has been dropped.
    pub async fn close(&self) -> Result<()> {
        let (ack_tx, ack_rx) = oneshot::channel();
        if self
            .inner
            .command_tx
            .send(EngineThreadCommand::Close(Some(ack_tx)))
            .is_err()
        {
            return Ok(());
        }
        match ack_rx.await {
            Ok(result) => result,
            Err(_) => Ok(()),
        }
    }

    /// Subscribe to engine lifecycle and request events.
    pub fn subscribe_events(&self) -> EngineEventReceiver {
        let (event_tx, event_rx) = mpsc::channel();
        if let Ok(mut subscribers) = self.inner.event_subscribers.lock() {
            subscribers.push(event_tx);
        }
        event_rx
    }
}

impl EngineLoad {
    fn spawn(model_path: PathBuf, config: NativeRuntimeConfig) -> Self {
        let model_id = model_path.display().to_string();
        let model_name = model_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(DEFAULT_MODEL_FILE_NAME)
            .to_string();
        let runtime_config = config;
        let (command_tx, command_rx) = mpsc::channel();
        let (init_tx, init_rx) = oneshot::channel();
        let event_subscribers = Arc::new(Mutex::new(Vec::new()));
        let engine_thread_subscribers = event_subscribers.clone();

        let join_handle = match thread::Builder::new()
            .name(ENGINE_THREAD_NAME.to_string())
            .spawn(move || {
                let runtime = InferenceRuntime::load(&model_path, runtime_config);
                match runtime {
                    Ok(runtime) => {
                        let model_state = ModelState {
                            id: model_id,
                            name: model_name,
                            capabilities: runtime.capabilities(),
                        };
                        let _ = init_tx.send(Ok(()));
                        run_engine_thread(
                            runtime,
                            command_rx,
                            model_state,
                            engine_thread_subscribers,
                        );
                    }
                    Err(error) => {
                        let _ = init_tx.send(Err(error));
                    }
                }
            }) {
            Ok(join_handle) => join_handle,
            Err(_) => {
                return Self {
                    command_tx: None,
                    init_rx: Some(ready_receiver(Err(runtime_command(
                        ENGINE_THREAD_SPAWN_FAILED,
                    )))),
                    event_subscribers,
                    join_handle: None,
                };
            }
        };
        Self {
            command_tx: Some(command_tx),
            init_rx: Some(init_rx),
            event_subscribers,
            join_handle: Some(join_handle),
        }
    }
}

impl Future for EngineLoad {
    type Output = Result<CogentEngine>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let Some(init_rx) = self.init_rx.as_mut() else {
            return Poll::Ready(Err(runtime_command("engine load future already resolved")));
        };
        match Pin::new(init_rx).poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(Ok(()))) => {
                self.init_rx.take();
                let command_tx = self
                    .command_tx
                    .take()
                    .ok_or_else(|| runtime_command(ENGINE_THREAD_CLOSED))?;
                let join_handle = self
                    .join_handle
                    .take()
                    .ok_or_else(|| runtime_command(ENGINE_THREAD_STOPPED_DURING_LOAD))?;
                Poll::Ready(Ok(CogentEngine {
                    inner: Arc::new(EngineInner {
                        command_tx,
                        event_subscribers: self.event_subscribers.clone(),
                        _driver: join_handle,
                    }),
                }))
            }
            Poll::Ready(Ok(Err(error))) => {
                self.init_rx.take();
                match join_load_thread(self.join_handle.take()) {
                    Ok(()) => Poll::Ready(Err(error)),
                    Err(error) => Poll::Ready(Err(error)),
                }
            }
            Poll::Ready(Err(_)) => {
                self.init_rx.take();
                match join_load_thread(self.join_handle.take()) {
                    Ok(()) => Poll::Ready(Err(runtime_command(ENGINE_THREAD_STOPPED_DURING_LOAD))),
                    Err(error) => Poll::Ready(Err(error)),
                }
            }
        }
    }
}

impl Drop for EngineLoad {
    fn drop(&mut self) {
        if self.init_rx.is_none() {
            return;
        }
        if let Some(command_tx) = self.command_tx.take() {
            let _ = command_tx.send(EngineThreadCommand::Close(None));
        }
        spawn_engine_load_reaper(self.join_handle.take());
    }
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        let _ = self.command_tx.send(EngineThreadCommand::Close(None));
    }
}

impl Future for EngineTextRun {
    type Output = Result<GenerationResult>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.response).poll(cx) {
            Poll::Ready(Ok(response)) => Poll::Ready(generation_result_from_response(response)),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl EngineTextRun {
    /// Borrow the token stream when streaming was requested for this run.
    pub fn token_stream(&mut self) -> Option<&mut EngineTokenStream> {
        self.tokens.as_mut()
    }

    /// Split the run into its optional token stream and final-response future.
    pub fn into_parts(self) -> (Option<EngineTokenStream>, EngineTextResponseFuture) {
        let Self {
            response,
            tokens,
            _engine,
        } = self;
        let future = Box::pin(async move {
            let _engine = _engine;
            generation_result_from_response(response.await?)
        });
        (tokens, future)
    }
}

impl Future for EngineEmbeddingRun {
    type Output = Result<EmbeddingResult>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match Pin::new(&mut self.response).poll(cx) {
            Poll::Ready(Ok(response)) => Poll::Ready(embedding_result_from_response(response)),
            Poll::Ready(Err(error)) => Poll::Ready(Err(error)),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl EngineEmbeddingRun {
    /// Convert the run into its final-response future.
    pub fn into_response(self) -> EngineEmbeddingResponseFuture {
        let Self { response, _engine } = self;
        Box::pin(async move {
            let _engine = _engine;
            embedding_result_from_response(response.await?)
        })
    }
}

impl Stream for EngineTokenStream {
    type Item = cogentlm_core::TokenBatch;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.rx).poll_next(cx)
    }
}

impl<T> Future for EngineResponse<T> {
    type Output = Result<T>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            Self::Pending(response_rx) => match Pin::new(response_rx).poll(cx) {
                Poll::Ready(Ok(result)) => Poll::Ready(result),
                Poll::Ready(Err(_)) => {
                    Poll::Ready(Err(runtime_command(ENGINE_THREAD_STOPPED_BEFORE_RESPONSE)))
                }
                Poll::Pending => Poll::Pending,
            },
            Self::Ready(result) => Poll::Ready(
                result
                    .take()
                    .unwrap_or_else(|| Err(runtime_command("engine response already consumed"))),
            ),
        }
    }
}

impl<T> EngineResponse<T> {
    fn ready_err(error: Error) -> Self {
        Self::Ready(Some(Err(error)))
    }
}

fn token_channel(
    enabled: bool,
) -> (
    Option<futures_mpsc::Sender<cogentlm_core::TokenBatch>>,
    Option<EngineTokenStream>,
) {
    if !enabled {
        return (None, None);
    }
    let (tx, rx) = futures_mpsc::channel(token_stream::TOKEN_STREAM_CHANNEL_CAPACITY);
    (Some(tx), Some(EngineTokenStream { rx }))
}

fn ready_receiver<T>(result: Result<T>) -> oneshot::Receiver<Result<T>> {
    let (tx, rx) = oneshot::channel();
    let _ = tx.send(result);
    rx
}

fn join_load_thread(join_handle: Option<JoinHandle<()>>) -> Result<()> {
    if let Some(join_handle) = join_handle {
        join_handle
            .join()
            .map_err(|_| runtime_command(ENGINE_THREAD_PANICKED))?;
    }
    Ok(())
}

fn spawn_engine_load_reaper(join_handle: Option<JoinHandle<()>>) {
    if let Some(join_handle) = join_handle {
        let _ = thread::Builder::new()
            .name("cogentlm-engine-load-reaper".to_string())
            .spawn(move || {
                let _ = join_handle.join();
            });
    }
}

fn runtime_command(message: impl Into<String>) -> Error {
    Error::RuntimeCommand(message.into())
}

#[cfg(test)]
mod tests {
    mod driver_tests;
}
