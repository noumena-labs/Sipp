//! Engine driver thread: owns the InferenceRuntime, accepts commands over a channel, and emits events.

use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::defaults::{BYTES_PER_KIB, DEFAULT_MODEL_FILE_NAME};
use crate::engine::{
    protocol::{EngineEvent, EngineState, ModelState, RequestResult},
    stream::TokenBatch,
    NativeRuntimeConfig,
};
use crate::error::{Error, Result};
use crate::runtime::request::GenerateResponse;
use crate::runtime::InferenceRuntime;

mod events;
mod request;
mod stats;
mod thread_loop;
mod token_sink;

pub use request::{ChatMessage, ChatRequest, ChatRole, QueryOptions, QueryRequest};
use stats::request_result_from_response;
use thread_loop::{run_engine_thread, EngineThreadCommand};

pub type QueryResponse = GenerateResponse;
pub type EngineEventReceiver = mpsc::Receiver<EngineEvent>;
type EngineEventSubscribers = Arc<Mutex<Vec<mpsc::Sender<EngineEvent>>>>;
pub(crate) type OnTokensCallback = Box<dyn FnMut(&TokenBatch) -> Result<()> + Send + 'static>;

const TOKEN_BATCH_MAX_FRAMES: usize = 64;
const TOKEN_BATCH_MAX_BYTES: usize = 64 * BYTES_PER_KIB;
const TOKEN_BATCH_FLUSH_INTERVAL: Duration = Duration::from_millis(4);
const ENGINE_CLOSE_ACK_TIMEOUT: Duration = Duration::from_secs(1);
const ENGINE_THREAD_CLOSED: &str = "engine thread is closed";
const ENGINE_THREAD_STOPPED_BEFORE_RESPONSE: &str = "engine thread stopped before responding";
const ENGINE_THREAD_NAME: &str = "cogentlm-engine";
const ENGINE_THREAD_SPAWN_FAILED: &str = "failed to spawn engine thread";
const ENGINE_THREAD_STOPPED_DURING_LOAD: &str = "engine thread stopped during load";
const ENGINE_THREAD_PANICKED: &str = "engine thread panicked";
const TOKEN_CALLBACK_THREAD_PANICKED: &str = "token callback thread panicked";

pub struct CogentEngine {
    command_tx: mpsc::Sender<EngineThreadCommand>,
    event_subscribers: EngineEventSubscribers,
    join_handle: Option<JoinHandle<()>>,
}

impl CogentEngine {
    pub fn load(model_path: impl AsRef<Path>, config: NativeRuntimeConfig) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        let model_state = ModelState {
            id: model_path.display().to_string(),
            name: model_path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(DEFAULT_MODEL_FILE_NAME)
                .to_string(),
        };
        let runtime_config = config;
        let (command_tx, command_rx) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::sync_channel(1);
        let event_subscribers = Arc::new(Mutex::new(Vec::new()));
        let engine_thread_subscribers = event_subscribers.clone();

        let join_handle = thread::Builder::new()
            .name(ENGINE_THREAD_NAME.to_string())
            .spawn(move || {
                let runtime = InferenceRuntime::load(&model_path, runtime_config);
                match runtime {
                    Ok(runtime) => {
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
            })
            .map_err(|_| runtime_command(ENGINE_THREAD_SPAWN_FAILED))?;

        match init_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                command_tx,
                event_subscribers,
                join_handle: Some(join_handle),
            }),
            Ok(Err(error)) => {
                let _ = join_handle.join();
                Err(error)
            }
            Err(_) => {
                let _ = join_handle.join();
                Err(runtime_command(ENGINE_THREAD_STOPPED_DURING_LOAD))
            }
        }
    }

    pub fn query(&self, request: impl Into<QueryRequest>) -> Result<RequestResult> {
        self.query_response(request)
            .map(|response| request_result_from_response(&response))
    }

    pub fn query_response(&self, request: impl Into<QueryRequest>) -> Result<QueryResponse> {
        self.send_command(|response_tx| EngineThreadCommand::Generate(request.into(), response_tx))
    }

    pub fn chat_response(&self, request: impl Into<ChatRequest>) -> Result<QueryResponse> {
        self.send_command(|response_tx| {
            EngineThreadCommand::GenerateChat(request.into(), response_tx)
        })
    }

    pub fn chat(&self, request: impl Into<ChatRequest>) -> Result<RequestResult> {
        self.chat_response(request)
            .map(|response| request_result_from_response(&response))
    }

    pub fn state(&self) -> Result<EngineState> {
        self.send_command(EngineThreadCommand::GetState)
    }

    pub fn subscribe_events(&self) -> EngineEventReceiver {
        let (event_tx, event_rx) = mpsc::channel();
        if let Ok(mut subscribers) = self.event_subscribers.lock() {
            subscribers.push(event_tx);
        }
        event_rx
    }

    pub fn close(mut self) -> Result<()> {
        self.close_inner()
    }

    fn close_inner(&mut self) -> Result<()> {
        let Some(join_handle) = self.join_handle.take() else {
            return Ok(());
        };

        let (ack_tx, ack_rx) = mpsc::channel();
        let send_result = self.command_tx.send(EngineThreadCommand::Close(ack_tx));
        if send_result.is_ok() {
            let _ = ack_rx.recv_timeout(ENGINE_CLOSE_ACK_TIMEOUT);
        }
        join_handle
            .join()
            .map_err(|_| runtime_command(ENGINE_THREAD_PANICKED))?;
        Ok(())
    }

    fn send_command<T>(
        &self,
        build_command: impl FnOnce(mpsc::Sender<Result<T>>) -> EngineThreadCommand,
    ) -> Result<T> {
        let (response_tx, response_rx) = mpsc::channel();
        self.command_tx
            .send(build_command(response_tx))
            .map_err(|_| runtime_command(ENGINE_THREAD_CLOSED))?;
        recv_command_response(response_rx)
    }
}

impl Drop for CogentEngine {
    fn drop(&mut self) {
        let _ = self.close_inner();
    }
}

fn recv_command_response<T>(response_rx: mpsc::Receiver<Result<T>>) -> Result<T> {
    response_rx
        .recv()
        .map_err(|_| runtime_command(ENGINE_THREAD_STOPPED_BEFORE_RESPONSE))?
}

fn runtime_command(message: impl Into<String>) -> Error {
    Error::RuntimeCommand(message.into())
}

#[cfg(test)]
mod tests {
    mod driver_tests;
}
