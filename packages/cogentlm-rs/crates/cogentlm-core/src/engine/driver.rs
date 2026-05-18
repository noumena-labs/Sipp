use std::path::Path;
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::runtime::request::{token_byte_ring, TOKEN_RING_DEFAULT_CAPACITY};

use serde_json::{json, Value};

use crate::engine::{
    protocol::{
        BackendDevice, BackendInfo, EngineEvent, EngineState, EngineStats, EngineStatus,
        FinishReason, ModelState, RequestResult, RequestStats,
    },
    stream::{StreamStats, TokenBatch},
    GenerateOptions, NativeRuntimeConfig, SamplingRuntimeConfig,
};
use crate::error::{Error, Result};
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::request::{
    GenerateResponse, GenerateResponseStatus, GenerateTokenEmissionMode, TokenByteRingConsumer,
    TokenByteRingProducer, TokenRingFrame,
};
use crate::runtime::{InferenceRuntime, RequestStepResult};

pub type QueryResponse = GenerateResponse;
pub type EngineEventReceiver = mpsc::Receiver<EngineEvent>;
type EngineEventSubscribers = Arc<Mutex<Vec<mpsc::Sender<EngineEvent>>>>;
type OnTokensCallback = Box<dyn FnMut(&TokenBatch) -> Result<()> + Send + 'static>;

const TOKEN_BATCH_MAX_FRAMES: usize = 64;
const TOKEN_BATCH_MAX_BYTES: usize = 64 * 1024;
const TOKEN_BATCH_FLUSH_INTERVAL: Duration = Duration::from_millis(4);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

impl ChatRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QueryOptions {
    pub context_key: String,
    pub max_tokens: i32,
    pub grammar: String,
    pub json_schema: String,
    pub stop: Vec<String>,
    pub sampling: Option<SamplingRuntimeConfig>,
    pub media: Vec<Vec<u8>>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            context_key: "default".to_string(),
            max_tokens: 64,
            grammar: String::new(),
            json_schema: String::new(),
            stop: Vec::new(),
            sampling: None,
            media: Vec::new(),
        }
    }
}

impl From<GenerateOptions> for QueryOptions {
    fn from(options: GenerateOptions) -> Self {
        Self {
            context_key: options.cache_key.unwrap_or_else(|| "default".to_string()),
            max_tokens: options.max_tokens,
            grammar: options.grammar.unwrap_or_default(),
            json_schema: options.json_schema.unwrap_or_default(),
            stop: options.stop,
            sampling: options.sampling,
            media: Vec::new(),
        }
    }
}

pub struct QueryRequest {
    pub prompt: String,
    pub options: QueryOptions,
    on_tokens: Option<OnTokensCallback>,
}

impl QueryRequest {
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            options: QueryOptions::default(),
            on_tokens: None,
        }
    }

    pub fn options(mut self, options: QueryOptions) -> Self {
        self.options = options;
        self
    }

    pub fn on_tokens(
        mut self,
        callback: impl FnMut(&TokenBatch) -> Result<()> + Send + 'static,
    ) -> Self {
        self.on_tokens = Some(Box::new(callback));
        self
    }
}

impl From<String> for QueryRequest {
    fn from(prompt: String) -> Self {
        Self::new(prompt)
    }
}

impl From<&str> for QueryRequest {
    fn from(prompt: &str) -> Self {
        Self::new(prompt)
    }
}

pub struct ChatRequest {
    pub messages: Vec<ChatMessage>,
    pub options: QueryOptions,
    on_tokens: Option<OnTokensCallback>,
}

impl ChatRequest {
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            options: QueryOptions::default(),
            on_tokens: None,
        }
    }

    pub fn options(mut self, options: QueryOptions) -> Self {
        self.options = options;
        self
    }

    pub fn on_tokens(
        mut self,
        callback: impl FnMut(&TokenBatch) -> Result<()> + Send + 'static,
    ) -> Self {
        self.on_tokens = Some(Box::new(callback));
        self
    }
}

impl From<Vec<ChatMessage>> for ChatRequest {
    fn from(messages: Vec<ChatMessage>) -> Self {
        Self::new(messages)
    }
}

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

    for command in command_rx {
        match command {
            EngineThreadCommand::Generate(request, response_tx) => {
                let response = if let Some(runtime) = runtime.as_mut() {
                    run_query(runtime, request, &model_state, &event_subscribers)
                } else {
                    Err(Error::RuntimeCommand("runtime is closed".to_string()))
                };
                let _ = response_tx.send(response);
            }
            EngineThreadCommand::GenerateChat(request, response_tx) => {
                let response = if let Some(runtime) = runtime.as_mut() {
                    run_chat(runtime, request, &model_state, &event_subscribers)
                } else {
                    Err(Error::RuntimeCommand("runtime is closed".to_string()))
                };
                let _ = response_tx.send(response);
            }
            EngineThreadCommand::GetState(response_tx) => {
                let response = if let Some(runtime) = runtime.as_ref() {
                    Ok(build_engine_state(runtime, &model_state))
                } else {
                    Err(Error::RuntimeCommand("runtime is closed".to_string()))
                };
                let _ = response_tx.send(response);
            }
            EngineThreadCommand::Close(ack_tx) => {
                drop(runtime.take());
                emit_event(&event_subscribers, EngineEvent::Closed);
                let _ = ack_tx.send(());
                break;
            }
        }
    }
}

fn build_engine_state(runtime: &InferenceRuntime, model_state: &ModelState) -> EngineState {
    build_engine_state_with_status(runtime, model_state, None)
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

fn engine_stats_from_runtime(metrics: RuntimeObservabilityMetrics) -> EngineStats {
    let tokens_per_second = if metrics.e2e_ms > 0.0 && metrics.output_tokens > 0 {
        Some(f64::from(metrics.output_tokens) / (metrics.e2e_ms / 1000.0))
    } else {
        None
    };
    let decode_tokens_per_second =
        decode_tokens_per_second(metrics.output_tokens, metrics.decode_ms);
    let prefill_tokens_per_second = if metrics.prefill_ms > 0.0 && metrics.prefill_tokens > 0 {
        Some(f64::from(metrics.prefill_tokens) / (metrics.prefill_ms / 1000.0))
    } else {
        None
    };

    EngineStats {
        input_tokens: i64::from(metrics.input_tokens),
        output_tokens: i64::from(metrics.output_tokens),
        cache_hits: i64::from(metrics.cache_hits),
        prefill_tokens: i64::from(metrics.prefill_tokens),
        ttft_ms: non_zero_metric(metrics.ttft_ms),
        inter_token_ms: non_zero_metric(metrics.itl_avg_ms),
        e2e_ms: non_zero_metric(metrics.e2e_ms),
        tokens_per_second,
        decode_tokens_per_second,
        prefill_tokens_per_second,
        prefill_ms: metrics.prefill_ms,
        decode_ms: metrics.decode_ms,
        backend_ms: metrics.native_gpu_ms,
        sync_ms: metrics.native_sync_ms,
        engine_overhead_ms: metrics.native_logic_ms,
        debug_metrics_scheduler_ticks: i64::from(metrics.debug_metrics_scheduler_ticks),
        debug_metrics_decode_ticks: i64::from(metrics.debug_metrics_decode_ticks),
        debug_metrics_prefill_ticks: i64::from(metrics.debug_metrics_prefill_ticks),
        debug_metrics_backend_sampler_attach_attempts: i64::from(
            metrics.debug_metrics_backend_sampler_attach_attempts,
        ),
        debug_metrics_backend_sampler_attach_failures: i64::from(
            metrics.debug_metrics_backend_sampler_attach_failures,
        ),
        debug_metrics_admit_ms: metrics.debug_metrics_admit_ms,
        debug_metrics_normalize_ms: metrics.debug_metrics_normalize_ms,
        debug_metrics_backend_sampler_attach_ms: metrics.debug_metrics_backend_sampler_attach_ms,
        debug_metrics_select_slots_ms: metrics.debug_metrics_select_slots_ms,
        debug_metrics_plan_ms: metrics.debug_metrics_plan_ms,
        debug_metrics_batch_build_ms: metrics.debug_metrics_batch_build_ms,
        debug_metrics_llama_decode_ms: metrics.debug_metrics_llama_decode_ms,
        debug_metrics_llama_sync_ms: metrics.debug_metrics_llama_sync_ms,
        debug_metrics_apply_bookkeeping_ms: metrics.debug_metrics_apply_bookkeeping_ms,
        debug_metrics_apply_decode_results_ms: metrics.debug_metrics_apply_decode_results_ms,
        debug_metrics_sample_ms: metrics.debug_metrics_sample_ms,
        debug_metrics_token_piece_ms: metrics.debug_metrics_token_piece_ms,
        debug_metrics_emit_ms: metrics.debug_metrics_emit_ms,
        debug_metrics_prefix_queue_ms: metrics.debug_metrics_prefix_queue_ms,
        debug_metrics_finalize_ms: metrics.debug_metrics_finalize_ms,
        debug_metrics_commit_observability_ms: metrics.debug_metrics_commit_observability_ms,
        debug_metrics_post_decode_ms: metrics.debug_metrics_post_decode_ms,
        ..EngineStats::default()
    }
}

fn non_zero_metric(value: f64) -> Option<f64> {
    (value > 0.0).then_some(value)
}

fn decode_tokens_per_second(output_tokens: i32, decode_ms: f64) -> Option<f64> {
    (output_tokens > 0 && decode_ms > 0.0).then(|| f64::from(output_tokens) / (decode_ms / 1000.0))
}

fn read_backend_info() -> BackendInfo {
    let Ok(raw) = crate::backend::backend_observability_json(true) else {
        return BackendInfo {
            selected: "unknown".to_string(),
            ..BackendInfo::default()
        };
    };
    let Ok(value) = serde_json::from_str::<Value>(&raw) else {
        return BackendInfo {
            selected: "unknown".to_string(),
            ..BackendInfo::default()
        };
    };

    let available = value
        .get("availableBackends")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("name").and_then(Value::as_str).map(str::to_string))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let selected = available
        .first()
        .cloned()
        .unwrap_or_else(|| "unknown".to_string());
    let devices = value
        .get("devices")
        .and_then(Value::as_array)
        .map(|items| items.iter().map(parse_backend_device).collect::<Vec<_>>())
        .unwrap_or_default();

    BackendInfo {
        selected,
        available,
        devices,
    }
}

fn parse_backend_device(value: &Value) -> BackendDevice {
    BackendDevice {
        id: value
            .get("deviceId")
            .and_then(Value::as_str)
            .map(str::to_string),
        name: value
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        device_type: value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_string(),
        memory_total_bytes: value.get("memoryTotalBytes").and_then(Value::as_u64),
        memory_free_bytes: value.get("memoryFreeBytes").and_then(Value::as_u64),
    }
}

fn unix_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn run_chat(
    runtime: &mut InferenceRuntime,
    request: ChatRequest,
    model_state: &ModelState,
    event_subscribers: &EngineEventSubscribers,
) -> Result<QueryResponse> {
    let prompt = render_chat_prompt(runtime, &request.messages)?;
    run_query(
        runtime,
        QueryRequest {
            prompt,
            options: request.options,
            on_tokens: request.on_tokens,
        },
        model_state,
        event_subscribers,
    )
}

fn run_query(
    runtime: &mut InferenceRuntime,
    request: QueryRequest,
    model_state: &ModelState,
    event_subscribers: &EngineEventSubscribers,
) -> Result<QueryResponse> {
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
    emit_state_event(
        runtime,
        model_state,
        event_subscribers,
        EngineStatus::Running,
    );

    drive_request_to_completion(
        runtime,
        request_id,
        on_tokens,
        model_state,
        event_subscribers,
    )
}

fn drive_request_to_completion(
    runtime: &mut InferenceRuntime,
    request_id: u32,
    on_tokens: Option<OnTokensCallback>,
    model_state: &ModelState,
    event_subscribers: &EngineEventSubscribers,
) -> Result<QueryResponse> {
    let mut token_sink = on_tokens.map(|callback| start_async_token_sink(request_id, callback));
    if let Some(sink) = &token_sink {
        runtime.set_token_ring_producer(Some(sink.producer.clone()));
    }

    loop {
        let burst = runtime.run_scheduler_loop(256, 1, 0, Duration::ZERO);

        if let Some(sink) = &mut token_sink {
            if let Some(error) = sink.try_recv_error() {
                runtime.set_token_ring_producer(None);
                sink.close();
                cancel_and_consume_request(runtime, request_id);
                let _ = sink.join();
                emit_event(
                    event_subscribers,
                    EngineEvent::RequestFailed {
                        request_id: request_id.to_string(),
                        error: error.to_string(),
                    },
                );
                emit_state_event(runtime, model_state, event_subscribers, EngineStatus::Ready);
                return Err(error);
            }
        }

        if let Some(response) = runtime.try_peek_completed_response(request_id) {
            let response = response.clone();
            runtime.consume_completed_response(request_id);
            runtime.set_token_ring_producer(None);
            if let Some(mut sink) = token_sink.take() {
                sink.close();
                sink.join()?;
            }
            let result = match response.status {
                GenerateResponseStatus::Completed => {
                    let result = request_result_from_response(&response);
                    emit_event(
                        event_subscribers,
                        EngineEvent::RequestCompleted {
                            result: result.clone(),
                        },
                    );
                    emit_state_event(runtime, model_state, event_subscribers, EngineStatus::Ready);
                    Ok(response)
                }
                GenerateResponseStatus::Cancelled => {
                    let message = response.error_message.if_empty("request cancelled");
                    emit_event(
                        event_subscribers,
                        EngineEvent::RequestFailed {
                            request_id: request_id.to_string(),
                            error: message.clone(),
                        },
                    );
                    emit_state_event(runtime, model_state, event_subscribers, EngineStatus::Ready);
                    Err(Error::RuntimeCommand(message))
                }
                GenerateResponseStatus::Failed => {
                    let message = response.error_message.if_empty("request failed");
                    emit_event(
                        event_subscribers,
                        EngineEvent::RequestFailed {
                            request_id: request_id.to_string(),
                            error: message.clone(),
                        },
                    );
                    emit_state_event(runtime, model_state, event_subscribers, EngineStatus::Ready);
                    Err(Error::RuntimeCommand(message))
                }
                GenerateResponseStatus::Pending => Err(Error::RuntimeCommand(
                    "request returned pending response".to_string(),
                )),
            };
            return result;
        }
        if matches!(
            burst.status,
            RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
        ) {
            runtime.set_token_ring_producer(None);
            if let Some(mut sink) = token_sink.take() {
                sink.close();
                let _ = sink.join();
            }
            let error = Error::RuntimeCommand(format!("scheduler stopped with {:?}", burst.status));
            emit_event(
                event_subscribers,
                EngineEvent::RequestFailed {
                    request_id: request_id.to_string(),
                    error: error.to_string(),
                },
            );
            emit_state_event(runtime, model_state, event_subscribers, EngineStatus::Ready);
            return Err(error);
        }
        if burst.status == RequestStepResult::Waiting && !runtime.has_request(request_id) {
            runtime.set_token_ring_producer(None);
            if let Some(mut sink) = token_sink.take() {
                sink.close();
                let _ = sink.join();
            }
            let error =
                Error::RuntimeCommand("scheduler is waiting but request disappeared".to_string());
            emit_event(
                event_subscribers,
                EngineEvent::RequestFailed {
                    request_id: request_id.to_string(),
                    error: error.to_string(),
                },
            );
            emit_state_event(runtime, model_state, event_subscribers, EngineStatus::Ready);
            return Err(error);
        }
        if burst.status == RequestStepResult::Waiting {
            runtime.set_token_ring_producer(None);
            if let Some(mut sink) = token_sink.take() {
                sink.close();
                let _ = sink.join();
            }
            let error = Error::RuntimeCommand(
                "scheduler is waiting while request is still live".to_string(),
            );
            emit_event(
                event_subscribers,
                EngineEvent::RequestFailed {
                    request_id: request_id.to_string(),
                    error: error.to_string(),
                },
            );
            emit_state_event(runtime, model_state, event_subscribers, EngineStatus::Ready);
            return Err(error);
        }
    }
}

struct AsyncTokenSink {
    producer: TokenByteRingProducer,
    join_handle: Option<JoinHandle<()>>,
    error_rx: mpsc::Receiver<Error>,
}

impl AsyncTokenSink {
    fn close(&self) {
        self.producer.close();
    }

    fn try_recv_error(&mut self) -> Option<Error> {
        match self.error_rx.try_recv() {
            Ok(error) => Some(error),
            Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => None,
        }
    }

    fn join(&mut self) -> Result<()> {
        if let Some(join_handle) = self.join_handle.take() {
            join_handle
                .join()
                .map_err(|_| Error::RuntimeCommand("token callback thread panicked".to_string()))?;
        }
        if let Some(error) = self.try_recv_error() {
            return Err(error);
        }
        Ok(())
    }
}

fn start_async_token_sink(request_id: u32, callback: OnTokensCallback) -> AsyncTokenSink {
    let (producer, consumer) = token_byte_ring(TOKEN_RING_DEFAULT_CAPACITY);
    let (error_tx, error_rx) = mpsc::channel();
    let join_handle = thread::spawn(move || {
        run_token_callback_loop(request_id, consumer, callback, error_tx);
    });
    AsyncTokenSink {
        producer,
        join_handle: Some(join_handle),
        error_rx,
    }
}

fn run_token_callback_loop(
    request_id: u32,
    consumer: TokenByteRingConsumer,
    mut callback: OnTokensCallback,
    error_tx: mpsc::Sender<Error>,
) {
    let mut token_state = TokenStreamState::new(request_id);
    let mut frames = Vec::with_capacity(TOKEN_BATCH_MAX_FRAMES);
    loop {
        consumer.wait_for_data(TOKEN_BATCH_FLUSH_INTERVAL);
        let batch_started = Instant::now();
        frames.clear();
        let mut latest_drop_count = token_state.last_drop_count;
        let mut closed = false;
        let mut byte_count = 0usize;

        loop {
            let remaining_frames = TOKEN_BATCH_MAX_FRAMES.saturating_sub(frames.len()).max(1);
            let remaining_bytes = TOKEN_BATCH_MAX_BYTES.saturating_sub(byte_count).max(1);
            let drain = consumer.drain_into(&mut frames, remaining_frames, remaining_bytes);
            latest_drop_count = latest_drop_count.max(drain.drop_count);
            closed |= drain.closed;
            byte_count = byte_count.saturating_add(drain.bytes_drained);

            if closed
                || frames.len() >= TOKEN_BATCH_MAX_FRAMES
                || byte_count >= TOKEN_BATCH_MAX_BYTES
                || batch_started.elapsed() >= TOKEN_BATCH_FLUSH_INTERVAL
            {
                break;
            }

            let remaining = TOKEN_BATCH_FLUSH_INTERVAL
                .checked_sub(batch_started.elapsed())
                .unwrap_or(Duration::ZERO);
            if remaining.is_zero() || !consumer.wait_for_data(remaining) {
                break;
            }
        }

        if let Some(batch) =
            token_batch_from_ring_frames(&frames, request_id, &mut token_state, latest_drop_count)
        {
            if let Err(error) = callback(&batch) {
                let _ = error_tx.send(error);
                return;
            }
        }

        if closed {
            return;
        }
    }
}

struct TokenStreamState {
    request_id: u32,
    next_sequence: u32,
    last_drop_count: u64,
    stats: StreamStats,
}

impl TokenStreamState {
    fn new(request_id: u32) -> Self {
        Self {
            request_id,
            next_sequence: 0,
            last_drop_count: 0,
            stats: StreamStats::default(),
        }
    }
}

fn token_batch_from_ring_frames(
    frames: &[TokenRingFrame],
    target_request_id: u32,
    token_state: &mut TokenStreamState,
    drop_count: u64,
) -> Option<TokenBatch> {
    let text_capacity = frames
        .iter()
        .filter(|frame| frame.stream_id == target_request_id)
        .map(|frame| frame.bytes.len())
        .sum();
    let mut text = String::with_capacity(text_capacity);
    let mut frame_count = 0_u32;
    let mut byte_count = 0_u32;
    let mut sequence_start = None;

    for frame in frames {
        if frame.stream_id != target_request_id {
            continue;
        }
        if sequence_start.is_none() {
            sequence_start = Some(frame.sequence);
        }
        match std::str::from_utf8(&frame.bytes) {
            Ok(piece) => text.push_str(piece),
            Err(_) => text.push_str(&String::from_utf8_lossy(&frame.bytes)),
        }
        frame_count = frame_count.saturating_add(1);
        byte_count = byte_count.saturating_add(frame.bytes.len() as u32);
    }

    update_stream_drop_stats(token_state, drop_count);

    if frame_count == 0 {
        return None;
    }

    token_state.next_sequence = sequence_start
        .unwrap_or(token_state.next_sequence)
        .saturating_add(frame_count);
    token_state.stats.frames_sent = token_state
        .stats
        .frames_sent
        .saturating_add(u64::from(frame_count));
    token_state.stats.bytes_sent = token_state
        .stats
        .bytes_sent
        .saturating_add(u64::from(byte_count));
    token_state.stats.batches_sent = token_state.stats.batches_sent.saturating_add(1);

    Some(TokenBatch {
        request_id: token_state.request_id.to_string(),
        stream_id: token_state.request_id,
        sequence_start: sequence_start.unwrap_or_default(),
        text,
        frame_count,
        byte_count,
        stats: token_state.stats,
    })
}

fn update_stream_drop_stats(token_state: &mut TokenStreamState, drop_count: u64) {
    let drop_delta = drop_count.saturating_sub(token_state.last_drop_count);
    token_state.last_drop_count = drop_count;
    token_state.stats.frames_dropped = token_state.stats.frames_dropped.saturating_add(drop_delta);
}

fn request_result_from_response(response: &GenerateResponse) -> RequestResult {
    RequestResult {
        id: response.request_id.to_string(),
        text: response.output_text.clone(),
        finish_reason: match response.status {
            GenerateResponseStatus::Completed => FinishReason::Stop,
            GenerateResponseStatus::Cancelled => FinishReason::Cancelled,
            GenerateResponseStatus::Failed => FinishReason::Error,
            GenerateResponseStatus::Pending => FinishReason::Error,
        },
        stats: request_stats_from_runtime(response.runtime_observability),
    }
}

fn request_stats_from_runtime(metrics: RuntimeObservabilityMetrics) -> RequestStats {
    let tokens_per_second = if metrics.e2e_ms > 0.0 && metrics.output_tokens > 0 {
        Some(f64::from(metrics.output_tokens) / (metrics.e2e_ms / 1000.0))
    } else {
        None
    };
    let decode_tokens_per_second =
        decode_tokens_per_second(metrics.output_tokens, metrics.decode_ms);

    RequestStats {
        input_tokens: metrics.input_tokens,
        output_tokens: metrics.output_tokens,
        cache_hits: metrics.cache_hits,
        ttft_ms: non_zero_metric(metrics.ttft_ms),
        inter_token_ms: non_zero_metric(metrics.itl_avg_ms),
        e2e_ms: non_zero_metric(metrics.e2e_ms),
        tokens_per_second,
        decode_tokens_per_second,
        prefill_ms: metrics.prefill_ms,
        decode_ms: metrics.decode_ms,
        debug_metrics_scheduler_ticks: metrics.debug_metrics_scheduler_ticks,
        debug_metrics_decode_ticks: metrics.debug_metrics_decode_ticks,
        debug_metrics_prefill_ticks: metrics.debug_metrics_prefill_ticks,
        debug_metrics_backend_sampler_attach_attempts: metrics
            .debug_metrics_backend_sampler_attach_attempts,
        debug_metrics_backend_sampler_attach_failures: metrics
            .debug_metrics_backend_sampler_attach_failures,
        debug_metrics_admit_ms: metrics.debug_metrics_admit_ms,
        debug_metrics_normalize_ms: metrics.debug_metrics_normalize_ms,
        debug_metrics_backend_sampler_attach_ms: metrics.debug_metrics_backend_sampler_attach_ms,
        debug_metrics_select_slots_ms: metrics.debug_metrics_select_slots_ms,
        debug_metrics_plan_ms: metrics.debug_metrics_plan_ms,
        debug_metrics_batch_build_ms: metrics.debug_metrics_batch_build_ms,
        debug_metrics_llama_decode_ms: metrics.debug_metrics_llama_decode_ms,
        debug_metrics_llama_sync_ms: metrics.debug_metrics_llama_sync_ms,
        debug_metrics_apply_bookkeeping_ms: metrics.debug_metrics_apply_bookkeeping_ms,
        debug_metrics_apply_decode_results_ms: metrics.debug_metrics_apply_decode_results_ms,
        debug_metrics_sample_ms: metrics.debug_metrics_sample_ms,
        debug_metrics_token_piece_ms: metrics.debug_metrics_token_piece_ms,
        debug_metrics_emit_ms: metrics.debug_metrics_emit_ms,
        debug_metrics_prefix_queue_ms: metrics.debug_metrics_prefix_queue_ms,
        debug_metrics_finalize_ms: metrics.debug_metrics_finalize_ms,
        debug_metrics_commit_observability_ms: metrics.debug_metrics_commit_observability_ms,
        debug_metrics_post_decode_ms: metrics.debug_metrics_post_decode_ms,
    }
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
    let rendered = messages
        .iter()
        .map(|message| {
            json!({
                "role": message.role.as_str(),
                "content": message.content,
            })
        })
        .collect::<Vec<_>>();
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
mod tests {
    use super::*;

    #[test]
    fn render_messages_json_uses_template_roles() {
        let messages = vec![
            ChatMessage::system("policy"),
            ChatMessage::user("hello"),
            ChatMessage::assistant("hi"),
        ];

        let rendered = render_messages_json(&messages).expect("json");

        assert_eq!(
            rendered,
            r#"[{"content":"policy","role":"system"},{"content":"hello","role":"user"},{"content":"hi","role":"assistant"}]"#
        );
    }

    #[test]
    fn query_options_default_matches_public_completion_defaults() {
        let options = QueryOptions::default();

        assert_eq!(options.context_key, "default");
        assert_eq!(options.max_tokens, 64);
        assert!(options.grammar.is_empty());
        assert!(options.json_schema.is_empty());
        assert!(options.stop.is_empty());
        assert!(options.sampling.is_none());
        assert!(options.media.is_empty());
    }

    #[test]
    fn generate_options_convert_to_query_options() {
        let options = QueryOptions::from(GenerateOptions {
            max_tokens: 7,
            stream: true,
            stop: vec!["END".to_string()],
            sampling: Some(SamplingRuntimeConfig {
                temperature: Some(0.1),
                ..SamplingRuntimeConfig::default()
            }),
            grammar: Some("root ::= \"x\"".to_string()),
            json_schema: Some("{}".to_string()),
            cache_key: Some("ctx".to_string()),
        });

        assert_eq!(options.context_key, "ctx");
        assert_eq!(options.max_tokens, 7);
        assert_eq!(options.grammar, "root ::= \"x\"");
        assert_eq!(options.json_schema, "{}");
        assert_eq!(options.stop, vec!["END"]);
        assert_eq!(
            options
                .sampling
                .as_ref()
                .and_then(|sampling| sampling.temperature),
            Some(0.1)
        );
    }

    #[test]
    fn query_request_defaults_options() {
        let request = QueryRequest::new("hello");

        assert_eq!(request.prompt, "hello");
        assert_eq!(request.options, QueryOptions::default());
    }

    #[test]
    fn token_ring_frames_are_batched_by_request() {
        let frames = vec![
            TokenRingFrame {
                stream_id: 1,
                sequence: 0,
                flags: 0,
                bytes: b"hel".to_vec(),
            },
            TokenRingFrame {
                stream_id: 2,
                sequence: 0,
                flags: 0,
                bytes: b"skip".to_vec(),
            },
            TokenRingFrame {
                stream_id: 1,
                sequence: 1,
                flags: 0,
                bytes: b"lo".to_vec(),
            },
        ];
        let mut state = TokenStreamState::new(1);

        let batch = token_batch_from_ring_frames(&frames, 1, &mut state, 0).expect("token batch");

        assert_eq!(batch.request_id, "1");
        assert_eq!(batch.stream_id, 1);
        assert_eq!(batch.sequence_start, 0);
        assert_eq!(batch.text, "hello");
        assert_eq!(batch.frame_count, 2);
        assert_eq!(batch.byte_count, 5);
        assert_eq!(batch.stats.frames_sent, 2);
        assert_eq!(batch.stats.bytes_sent, 5);
        assert_eq!(batch.stats.batches_sent, 1);
    }

    #[test]
    fn token_ring_batch_tracks_drops_and_sequences() {
        let first = [TokenRingFrame {
            stream_id: 3,
            sequence: 0,
            flags: 0,
            bytes: b"a".to_vec(),
        }];
        let second = [TokenRingFrame {
            stream_id: 3,
            sequence: 1,
            flags: 0,
            bytes: b"bc".to_vec(),
        }];
        let mut state = TokenStreamState::new(3);

        let first = token_batch_from_ring_frames(&first, 3, &mut state, 2).expect("first batch");
        let second = token_batch_from_ring_frames(&second, 3, &mut state, 5).expect("second batch");

        assert_eq!(first.sequence_start, 0);
        assert_eq!(first.stats.frames_dropped, 2);
        assert_eq!(second.sequence_start, 1);
        assert_eq!(second.stats.frames_sent, 2);
        assert_eq!(second.stats.frames_dropped, 5);
        assert_eq!(second.stats.bytes_sent, 3);
        assert_eq!(second.stats.batches_sent, 2);
    }

    #[test]
    fn runtime_metrics_map_to_engine_stats() {
        let stats = engine_stats_from_runtime(RuntimeObservabilityMetrics {
            ttft_ms: 10.0,
            itl_avg_ms: 5.0,
            e2e_ms: 100.0,
            prefill_ms: 25.0,
            decode_ms: 75.0,
            native_gpu_ms: 60.0,
            native_sync_ms: 15.0,
            native_logic_ms: 2.0,
            input_tokens: 8,
            output_tokens: 4,
            cache_hits: 3,
            prefill_tokens: 5,
            ..RuntimeObservabilityMetrics::default()
        });

        assert_eq!(stats.input_tokens, 8);
        assert_eq!(stats.output_tokens, 4);
        assert_eq!(stats.cache_hits, 3);
        assert_eq!(stats.prefill_tokens, 5);
        assert_eq!(stats.ttft_ms, Some(10.0));
        assert_eq!(stats.inter_token_ms, Some(5.0));
        assert_eq!(stats.e2e_ms, Some(100.0));
        assert_eq!(stats.tokens_per_second, Some(40.0));
        assert_eq!(stats.prefill_tokens_per_second, Some(200.0));
        assert_eq!(stats.backend_ms, 60.0);
        assert_eq!(stats.sync_ms, 15.0);
        assert_eq!(stats.engine_overhead_ms, 2.0);
    }

    #[test]
    fn completed_response_maps_to_request_result() {
        let result = request_result_from_response(&GenerateResponse {
            request_id: 7,
            status: GenerateResponseStatus::Completed,
            output_text: "hello".to_string(),
            runtime_observability: RuntimeObservabilityMetrics {
                e2e_ms: 50.0,
                output_tokens: 5,
                ..RuntimeObservabilityMetrics::default()
            },
            ..GenerateResponse::default()
        });

        assert_eq!(result.id, "7");
        assert_eq!(result.text, "hello");
        assert_eq!(result.finish_reason, FinishReason::Stop);
        assert_eq!(result.stats.output_tokens, 5);
        assert_eq!(result.stats.tokens_per_second, Some(100.0));
    }

    #[test]
    fn emit_event_drops_closed_subscribers() {
        let subscribers = Arc::new(Mutex::new(Vec::new()));
        let (closed_tx, closed_rx) = mpsc::channel();
        drop(closed_rx);
        let (open_tx, open_rx) = mpsc::channel();
        subscribers.lock().unwrap().push(closed_tx);
        subscribers.lock().unwrap().push(open_tx);

        emit_event(&subscribers, EngineEvent::Closed);

        assert!(matches!(open_rx.recv().unwrap(), EngineEvent::Closed));
        assert_eq!(subscribers.lock().unwrap().len(), 1);
    }

    #[test]
    fn engine_handle_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CogentEngine>();
    }
}
