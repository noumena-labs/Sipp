//! Engine driver thread: owns the InferenceRuntime, accepts commands over a channel, and emits events.

use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use serde_json::json;

use crate::engine::{
    protocol::{EngineEvent, EngineState, EngineStatus, ModelState, RequestResult},
    stream::TokenBatch,
    NativeRuntimeConfig,
};
use crate::error::{Error, Result};
use crate::runtime::request::{GenerateResponse, GenerateResponseStatus, GenerateTokenEmissionMode};
use crate::runtime::{InferenceRuntime, RequestStepResult};

mod request;
mod stats;
mod token_sink;

pub use request::{ChatMessage, ChatRequest, ChatRole, QueryOptions, QueryRequest};
use stats::{
    engine_stats_from_runtime, read_backend_info, request_result_from_response, unix_time_ms,
};
use token_sink::{start_async_token_sink, AsyncTokenSink};

pub type QueryResponse = GenerateResponse;
pub type EngineEventReceiver = mpsc::Receiver<EngineEvent>;
type EngineEventSubscribers = Arc<Mutex<Vec<mpsc::Sender<EngineEvent>>>>;
pub(crate) type OnTokensCallback = Box<dyn FnMut(&TokenBatch) -> Result<()> + Send + 'static>;

const TOKEN_BATCH_MAX_FRAMES: usize = 64;
const TOKEN_BATCH_MAX_BYTES: usize = 64 * 1024;
const TOKEN_BATCH_FLUSH_INTERVAL: Duration = Duration::from_millis(4);

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
                .unwrap_or("model.gguf")
                .to_string(),
        };
        let runtime_config = config;
        let (command_tx, command_rx) = mpsc::channel();
        let (init_tx, init_rx) = mpsc::sync_channel(1);
        let event_subscribers = Arc::new(Mutex::new(Vec::new()));
        let engine_thread_subscribers = event_subscribers.clone();

        let join_handle = thread::Builder::new()
            .name("cogentlm-engine".to_string())
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
            .map_err(|_| Error::RuntimeCommand("failed to spawn engine thread".to_string()))?;

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
                Err(Error::RuntimeCommand(
                    "engine thread stopped during load".to_string(),
                ))
            }
        }
    }

    pub fn query(&self, request: impl Into<QueryRequest>) -> Result<RequestResult> {
        self.query_response(request)
            .map(|response| request_result_from_response(&response))
    }

    pub fn query_response(&self, request: impl Into<QueryRequest>) -> Result<QueryResponse> {
        let (response_tx, response_rx) = mpsc::channel();
        self.command_tx
            .send(EngineThreadCommand::Generate(request.into(), response_tx))
            .map_err(|_| Error::RuntimeCommand("engine thread is closed".to_string()))?;
        recv_command_response(response_rx)
    }

    pub fn chat_response(&self, request: impl Into<ChatRequest>) -> Result<QueryResponse> {
        let (response_tx, response_rx) = mpsc::channel();
        self.command_tx
            .send(EngineThreadCommand::GenerateChat(
                request.into(),
                response_tx,
            ))
            .map_err(|_| Error::RuntimeCommand("engine thread is closed".to_string()))?;
        recv_command_response(response_rx)
    }

    pub fn chat(&self, request: impl Into<ChatRequest>) -> Result<RequestResult> {
        self.chat_response(request)
            .map(|response| request_result_from_response(&response))
    }

    pub fn state(&self) -> Result<EngineState> {
        let (response_tx, response_rx) = mpsc::channel();
        self.command_tx
            .send(EngineThreadCommand::GetState(response_tx))
            .map_err(|_| Error::RuntimeCommand("engine thread is closed".to_string()))?;
        recv_command_response(response_rx)
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
            let _ = ack_rx.recv_timeout(Duration::from_secs(1));
        }
        join_handle
            .join()
            .map_err(|_| Error::RuntimeCommand("engine thread panicked".to_string()))?;
        Ok(())
    }
}

impl Drop for CogentEngine {
    fn drop(&mut self) {
        let _ = self.close_inner();
    }
}

enum EngineThreadCommand {
    Generate(QueryRequest, mpsc::Sender<Result<QueryResponse>>),
    GenerateChat(ChatRequest, mpsc::Sender<Result<QueryResponse>>),
    GetState(mpsc::Sender<Result<EngineState>>),
    Close(mpsc::Sender<()>),
}

fn run_engine_thread(
    runtime: InferenceRuntime,
    command_rx: mpsc::Receiver<EngineThreadCommand>,
    model_state: ModelState,
    event_subscribers: EngineEventSubscribers,
) {
    let mut runtime = Some(runtime);
    let mut active_requests: std::collections::HashMap<u32, mpsc::Sender<Result<QueryResponse>>> = std::collections::HashMap::new();
    let mut token_sinks: std::collections::HashMap<u32, AsyncTokenSink> = std::collections::HashMap::new();

    loop {
        if active_requests.is_empty() {
            let Ok(command) = command_rx.recv() else {
                break;
            };
            if !process_command(
                command,
                &mut runtime,
                &mut active_requests,
                &mut token_sinks,
                &model_state,
                &event_subscribers,
            ) {
                break;
            }
        } else {
            let mut stop = false;
            while let Ok(command) = command_rx.try_recv() {
                if !process_command(
                    command,
                    &mut runtime,
                    &mut active_requests,
                    &mut token_sinks,
                    &model_state,
                    &event_subscribers,
                ) {
                    stop = true;
                    break;
                }
            }
            if stop {
                break;
            }

            if let Some(runtime_mut) = runtime.as_mut() {
                if !active_requests.is_empty() {
                    let burst = runtime_mut.run_scheduler_loop(1, 0, 0, Duration::ZERO);

                    let mut errored_ids = Vec::new();
                    for (&request_id, sink) in token_sinks.iter_mut() {
                        if let Some(error) = sink.try_recv_error() {
                            errored_ids.push((request_id, error));
                        }
                    }
                    for (request_id, error) in errored_ids {
                        cancel_and_consume_request(runtime_mut, request_id);
                        let error_msg = error.to_string();
                        if let Some(response_tx) = active_requests.remove(&request_id) {
                            let _ = response_tx.send(Err(error));
                        }
                        if let Some(mut sink) = token_sinks.remove(&request_id) {
                            sink.close();
                            let _ = sink.join();
                        }
                        runtime_mut.remove_token_ring_producer(request_id);
                        emit_event(
                            &event_subscribers,
                            EngineEvent::RequestFailed {
                                request_id: request_id.to_string(),
                                error: error_msg,
                            },
                        );
                    }

                    let mut completed_ids = Vec::new();
                    for (&request_id, _) in active_requests.iter() {
                        if let Some(response) = runtime_mut.take_completed_response(request_id) {
                            completed_ids.push((request_id, response));
                        }
                    }
                    for (request_id, response) in completed_ids {
                        if let Some(mut sink) = token_sinks.remove(&request_id) {
                            sink.close();
                            let _ = sink.join();
                        }
                        runtime_mut.remove_token_ring_producer(request_id);
                        
                        let response_tx = active_requests.remove(&request_id);
                        let result = match response.status {
                            GenerateResponseStatus::Completed => {
                                let result = request_result_from_response(&response);
                                emit_event(
                                    &event_subscribers,
                                    EngineEvent::RequestCompleted {
                                        result: result.clone(),
                                    },
                                );
                                Ok(response)
                            }
                            GenerateResponseStatus::Cancelled => {
                                let message = response.error_message.if_empty("request cancelled");
                                emit_event(
                                    &event_subscribers,
                                    EngineEvent::RequestFailed {
                                        request_id: request_id.to_string(),
                                        error: message.clone(),
                                    },
                                );
                                Err(Error::RuntimeCommand(message))
                            }
                            GenerateResponseStatus::Failed => {
                                let message = response.error_message.if_empty("request failed");
                                emit_event(
                                    &event_subscribers,
                                    EngineEvent::RequestFailed {
                                        request_id: request_id.to_string(),
                                        error: message.clone(),
                                    },
                                );
                                Err(Error::RuntimeCommand(message))
                            }
                            GenerateResponseStatus::Pending => Err(Error::RuntimeCommand(
                                "request returned pending response".to_string(),
                            )),
                        };

                        if let Some(tx) = response_tx {
                            let _ = tx.send(result);
                        }
                    }

                    if matches!(
                        burst.status,
                        RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
                    ) {
                        let error_msg = if burst.status == RequestStepResult::Invalid {
                            "Engine became invalid during execution.".to_string()
                        } else {
                            "Engine execution failed with no progress.".to_string()
                        };
                        
                        let remaining_ids: Vec<u32> = active_requests.keys().copied().collect();
                        for request_id in remaining_ids {
                            cancel_and_consume_request(runtime_mut, request_id);
                            if let Some(response_tx) = active_requests.remove(&request_id) {
                                let _ = response_tx.send(Err(Error::RuntimeCommand(error_msg.clone())));
                            }
                            if let Some(mut sink) = token_sinks.remove(&request_id) {
                                sink.close();
                                let _ = sink.join();
                            }
                            runtime_mut.remove_token_ring_producer(request_id);
                            emit_event(
                                &event_subscribers,
                                EngineEvent::RequestFailed {
                                    request_id: request_id.to_string(),
                                    error: error_msg.clone(),
                                },
                            );
                        }
                    }

                    if active_requests.is_empty() {
                        emit_state_event(runtime_mut, &model_state, &event_subscribers, EngineStatus::Ready);
                    }
                }
            }
        }
    }
}

fn process_command(
    command: EngineThreadCommand,
    runtime: &mut Option<InferenceRuntime>,
    active_requests: &mut std::collections::HashMap<u32, mpsc::Sender<Result<QueryResponse>>>,
    token_sinks: &mut std::collections::HashMap<u32, AsyncTokenSink>,
    model_state: &ModelState,
    event_subscribers: &EngineEventSubscribers,
) -> bool {
    match command {
        EngineThreadCommand::Generate(request, response_tx) => {
            if let Some(runtime_mut) = runtime.as_mut() {
                match start_query(runtime_mut, request, event_subscribers) {
                    Ok((request_id, token_sink)) => {
                        active_requests.insert(request_id, response_tx);
                        if let Some(sink) = token_sink {
                            token_sinks.insert(request_id, sink);
                        }
                        emit_state_event(runtime_mut, model_state, event_subscribers, EngineStatus::Running);
                    }
                    Err(error) => {
                        let _ = response_tx.send(Err(error));
                    }
                }
            } else {
                let _ = response_tx.send(Err(Error::RuntimeCommand("runtime is closed".to_string())));
            }
        }
        EngineThreadCommand::GenerateChat(request, response_tx) => {
            if let Some(runtime_mut) = runtime.as_mut() {
                match start_chat(runtime_mut, request, event_subscribers) {
                    Ok((request_id, token_sink)) => {
                        active_requests.insert(request_id, response_tx);
                        if let Some(sink) = token_sink {
                            token_sinks.insert(request_id, sink);
                        }
                        emit_state_event(runtime_mut, model_state, event_subscribers, EngineStatus::Running);
                    }
                    Err(error) => {
                        let _ = response_tx.send(Err(error));
                    }
                }
            } else {
                let _ = response_tx.send(Err(Error::RuntimeCommand("runtime is closed".to_string())));
            }
        }
        EngineThreadCommand::GetState(response_tx) => {
            let response = if let Some(runtime_ref) = runtime.as_ref() {
                let status = if active_requests.is_empty() {
                    EngineStatus::Ready
                } else {
                    EngineStatus::Running
                };
                Ok(build_engine_state_with_status(runtime_ref, model_state, Some(status)))
            } else {
                Err(Error::RuntimeCommand("runtime is closed".to_string()))
            };
            let _ = response_tx.send(response);
        }
        EngineThreadCommand::Close(ack_tx) => {
            if let Some(runtime_mut) = runtime.as_mut() {
                let remaining_ids: Vec<u32> = active_requests.keys().copied().collect();
                for request_id in remaining_ids {
                    cancel_and_consume_request(runtime_mut, request_id);
                    if let Some(response_tx) = active_requests.remove(&request_id) {
                        let _ = response_tx.send(Err(Error::RuntimeCommand("engine closed".to_string())));
                    }
                    if let Some(mut sink) = token_sinks.remove(&request_id) {
                        sink.close();
                        let _ = sink.join();
                    }
                    runtime_mut.remove_token_ring_producer(request_id);
                }
            }
            drop(runtime.take());
            emit_event(event_subscribers, EngineEvent::Closed);
            let _ = ack_tx.send(());
            return false;
        }
    }
    true
}


fn build_engine_state_with_status(
    runtime: &InferenceRuntime,
    model_state: &ModelState,
    status: Option<EngineStatus>,
) -> EngineState {
    EngineState {
        status: status.unwrap_or_else(|| {
            if runtime.is_ready() {
                EngineStatus::Ready
            } else {
                EngineStatus::Error
            }
        }),
        model: Some(model_state.clone()),
        backend: read_backend_info(),
        runtime: Some(runtime.resolved_runtime_limits()),
        requests: Vec::new(),
        stats: runtime
            .try_get_runtime_observability()
            .map(engine_stats_from_runtime)
            .unwrap_or_default(),
        updated_at_unix_ms: unix_time_ms(),
    }
}

fn emit_state_event(
    runtime: &InferenceRuntime,
    model_state: &ModelState,
    event_subscribers: &EngineEventSubscribers,
    status: EngineStatus,
) {
    emit_event(
        event_subscribers,
        EngineEvent::State(build_engine_state_with_status(
            runtime,
            model_state,
            Some(status),
        )),
    );
}

fn emit_event(event_subscribers: &EngineEventSubscribers, event: EngineEvent) {
    let Ok(mut subscribers) = event_subscribers.lock() else {
        return;
    };
    subscribers.retain(|subscriber| subscriber.send(event.clone()).is_ok());
}


fn start_chat(
    runtime: &mut InferenceRuntime,
    request: ChatRequest,
    event_subscribers: &EngineEventSubscribers,
) -> Result<(u32, Option<AsyncTokenSink>)> {
    let prompt = render_chat_prompt(runtime, &request.messages)?;
    start_query(
        runtime,
        QueryRequest {
            prompt,
            options: request.options,
            on_tokens: request.on_tokens,
        },
        event_subscribers,
    )
}

fn start_query(
    runtime: &mut InferenceRuntime,
    request: QueryRequest,
    event_subscribers: &EngineEventSubscribers,
) -> Result<(u32, Option<AsyncTokenSink>)> {
    let QueryRequest {
        prompt,
        options,
        on_tokens,
    } = request;

    if options.max_tokens <= 0 {
        return Err(Error::InvalidRequest("max_tokens must be positive"));
    }

    let emit_tokens = on_tokens.is_some();
    let emission_mode = if emit_tokens {
        GenerateTokenEmissionMode::TokenStream
    } else {
        GenerateTokenEmissionMode::None
    };

    let request_id = if options.media.is_empty() {
        runtime.enqueue_request(
            options.context_key,
            prompt,
            options.max_tokens,
            options.grammar,
            options.json_schema,
            options.stop,
            options.sampling,
            emission_mode,
        )?
    } else {
        runtime.enqueue_multimodal_request(
            options.context_key,
            prompt,
            options.max_tokens,
            options.media,
            options.grammar,
            options.json_schema,
            options.stop,
            options.sampling,
            emission_mode,
        )?
    };

    emit_event(
        event_subscribers,
        EngineEvent::RequestStarted {
            request_id: request_id.to_string(),
            stream_id: request_id,
        },
    );

    let token_sink = on_tokens.map(|callback| start_async_token_sink(request_id, callback));
    if let Some(sink) = &token_sink {
        runtime.add_token_ring_producer(request_id, sink.producer.clone());
    }

    Ok((request_id, token_sink))
}



fn cancel_and_consume_request(runtime: &mut InferenceRuntime, request_id: u32) {
    let _ = runtime.cancel_request(request_id);

    loop {
        if runtime.try_peek_completed_response(request_id).is_some() {
            runtime.consume_completed_response(request_id);
            return;
        }
        if !runtime.has_request(request_id) {
            return;
        }

        let burst = runtime.run_scheduler_loop(256, 1, 0, Duration::ZERO);

        if matches!(
            burst.status,
            RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
        ) {
            return;
        }
        if burst.status == RequestStepResult::Waiting
            && runtime.try_peek_completed_response(request_id).is_none()
        {
            return;
        }
    }
}

fn render_chat_prompt(runtime: &InferenceRuntime, messages: &[ChatMessage]) -> Result<String> {
    if messages.is_empty() {
        return Err(Error::InvalidRequest("chat messages must not be empty"));
    }
    let messages_json = render_messages_json(messages)?;
    let prompt = runtime.apply_chat_template_json(&messages_json, true)?;
    if prompt.is_empty() {
        return Err(Error::RuntimeCommand(
            "model chat template did not produce a prompt".to_string(),
        ));
    }
    Ok(prompt)
}

fn render_messages_json(messages: &[ChatMessage]) -> Result<String> {
    let mut rendered = Vec::with_capacity(messages.len());
    rendered.extend(messages.iter().map(|message| {
        json!({
            "role": message.role.as_str(),
            "content": message.content,
        })
    }));
    serde_json::to_string(&rendered)
        .map_err(|error| Error::RuntimeCommand(format!("failed to render chat JSON: {error}")))
}

fn recv_command_response<T>(response_rx: mpsc::Receiver<Result<T>>) -> Result<T> {
    response_rx
        .recv()
        .map_err(|_| Error::RuntimeCommand("engine thread stopped before responding".to_string()))?
}

trait EmptyStringFallback {
    fn if_empty(self, fallback: &'static str) -> String;
}

impl EmptyStringFallback for String {
    fn if_empty(self, fallback: &'static str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests;
