use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(target_family = "wasm")]
use std::time::{Duration, Instant};

#[cfg(target_family = "wasm")]
use cogentlm_engine::backend::set_llama_log_quiet;
#[cfg(target_family = "wasm")]
use cogentlm_engine::engine::protocol::{EmbedOptions, PoolingType};
#[cfg(target_family = "wasm")]
use cogentlm_engine::runtime::config::NativeRuntimeConfig;
#[cfg(target_family = "wasm")]
use cogentlm_engine::runtime::request::{
    token_byte_ring, GenerateResponse, GenerateResponseStatus, GenerateTokenEmissionMode,
    ResponseOutput, TokenByteRingConsumer, TokenByteRingProducer, TOKEN_RING_DEFAULT_CAPACITY,
};
#[cfg(target_family = "wasm")]
use cogentlm_engine::runtime::{InferenceRuntime, RequestStepResult, SchedulerBurstResult};

use crate::{BrowserRuntimeMetrics, BrowserSchedulerLoopResult};

// Provided by the Emscripten JS library. Calls back into the host JS to drain
// the streaming buffer into the SAB ring. Synchronous; the worker thread
// blocks inside it for a few microseconds.
#[cfg(target_family = "wasm")]
extern "C" {
    fn ce_native_yield();
}

// Upper bound on ticks per burst in streaming mode. We yield as soon as a
// token is emitted (`max_emitted=1` per burst), so this only caps how long
// we'll spin through prefill ticks before letting the host drain an empty
// buffer. Larger values lower outer-loop overhead during long prompts;
// smaller values lower cancellation latency.
#[cfg(target_family = "wasm")]
const STREAMING_STEP_TICKS: i32 = 256;

pub(crate) const ABI_VERSION: u32 = 5;

#[cfg(target_family = "wasm")]
const STATUS_OK: i32 = 0;
const STATUS_FAILURE: i32 = -1;
const STATUS_INVALID_ARGUMENTS: i32 = -2;
#[cfg(not(target_family = "wasm"))]
const STATUS_UNAVAILABLE: i32 = -4;
#[cfg(target_family = "wasm")]
const STATUS_NOT_INITIALIZED: i32 = -5;

#[cfg(target_family = "wasm")]
const COMPLETED_REQUEST_STATUS_PENDING: i32 = 0;
#[cfg(target_family = "wasm")]
const COMPLETED_REQUEST_STATUS_COMPLETED: i32 = 1;
#[cfg(target_family = "wasm")]
const COMPLETED_REQUEST_STATUS_CANCELLED: i32 = 2;
#[cfg(target_family = "wasm")]
const COMPLETED_REQUEST_STATUS_FAILED: i32 = 3;
const COMPLETED_REQUEST_STATUS_UNKNOWN: i32 = 4;
#[cfg(target_family = "wasm")]
const COMPLETED_REQUEST_OUTPUT_TEXT: i32 = 1;
#[cfg(target_family = "wasm")]
const COMPLETED_REQUEST_OUTPUT_EMBEDDING: i32 = 2;

#[cfg(target_family = "wasm")]
const STREAMING_BUFFER_CAPACITY: usize = 256 * 1024;
#[cfg(target_family = "wasm")]
const STREAMING_RECORD_HEADER_BYTES: usize = 8;

static NEXT_ENGINE_ID: AtomicU32 = AtomicU32::new(1);

#[repr(C)]
pub struct BrowserEngine {
    id: u32,
    last_error: String,
    inner: BrowserEngineInner,
}

#[cfg(target_family = "wasm")]
struct BrowserEngineInner {
    runtime: Option<InferenceRuntime>,
    token_producer: Option<TokenByteRingProducer>,
    token_consumer: Option<TokenByteRingConsumer>,
    token_ring_drop_count: u64,
    streaming_buffer: StreamingBuffer,
}

#[cfg(not(target_family = "wasm"))]
struct BrowserEngineInner;

impl BrowserEngine {
    pub(crate) fn create() -> Self {
        let id = NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed);
        Self::new(id)
    }

    fn new(id: u32) -> Self {
        Self {
            id,
            last_error: String::new(),
            inner: BrowserEngineInner::new(),
        }
    }

    pub(crate) fn id(&self) -> u32 {
        self.id
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn load(&mut self, model_path: &str, runtime_config_json: &str) -> i32 {
        self.clear_last_error();

        let runtime_config = match read_runtime_config(runtime_config_json) {
            Ok(config) => config,
            Err(error) => {
                self.set_last_error(error);
                return STATUS_INVALID_ARGUMENTS;
            }
        };

        set_llama_log_quiet(true);
        self.close_runtime();

        let runtime = match InferenceRuntime::load(model_path, runtime_config) {
            Ok(runtime) => runtime,
            Err(error) => {
                self.set_last_error(format!("failed to load browser runtime: {error:#}"));
                return STATUS_FAILURE;
            }
        };
        let (producer, consumer) = token_byte_ring(TOKEN_RING_DEFAULT_CAPACITY);
        self.inner.runtime = Some(runtime);
        self.inner.token_producer = Some(producer);
        self.inner.token_consumer = Some(consumer);
        STATUS_OK
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn load(&mut self, _model_path: &str, _runtime_config_json: &str) -> i32 {
        self.set_last_error("browser runtime is unavailable for this target");
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    fn clear_last_error(&mut self) {
        self.last_error.clear();
    }

    fn set_last_error(&mut self, error: impl Into<String>) {
        self.last_error = error.into();
    }

    pub(crate) fn last_error(&self) -> &str {
        &self.last_error
    }

    #[cfg(target_family = "wasm")]
    fn close_runtime(&mut self) {
        self.inner.runtime = None;
        self.inner.token_producer = None;
        self.inner.token_consumer = None;
        self.inner.streaming_buffer.reset();
        self.inner.token_ring_drop_count = 0;
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn start_text_request(
        &mut self,
        context_key: &str,
        prompt: &str,
        max_tokens: i32,
        token_emission_mode: i32,
        grammar: &str,
    ) -> u32 {
        self.clear_last_error();
        self.enqueue_prompt_request(
            context_key.to_string(),
            prompt.to_string(),
            max_tokens,
            Vec::new(),
            grammar.to_string(),
            token_emission_mode,
        )
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn start_text_request(
        &mut self,
        _context_key: &str,
        _prompt: &str,
        _max_tokens: i32,
        _token_emission_mode: i32,
        _grammar: &str,
    ) -> u32 {
        0
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn start_media_request(
        &mut self,
        context_key: &str,
        prompt: &str,
        max_tokens: i32,
        images_flat_buffer: &[u8],
        image_sizes: &[i32],
        token_emission_mode: i32,
        grammar: &str,
    ) -> u32 {
        self.clear_last_error();
        let Some(images) = copy_image_buffers(images_flat_buffer, image_sizes) else {
            self.set_last_error("media buffers are invalid");
            return 0;
        };

        self.enqueue_prompt_request(
            context_key.to_string(),
            prompt.to_string(),
            max_tokens,
            images,
            grammar.to_string(),
            token_emission_mode,
        )
    }

    #[cfg(not(target_family = "wasm"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_media_request(
        &mut self,
        _context_key: &str,
        _prompt: &str,
        _max_tokens: i32,
        _images_flat_buffer: &[u8],
        _image_sizes: &[i32],
        _token_emission_mode: i32,
        _grammar: &str,
    ) -> u32 {
        0
    }

    #[cfg(target_family = "wasm")]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_chat_request(
        &mut self,
        context_key: &str,
        messages_json: &str,
        max_tokens: i32,
        images_flat_buffer: &[u8],
        image_sizes: &[i32],
        token_emission_mode: i32,
        grammar: &str,
    ) -> u32 {
        self.clear_last_error();
        let Some(runtime) = self.inner.runtime.as_ref() else {
            self.set_last_error("runtime is not loaded");
            return 0;
        };
        let Ok(prompt) = runtime.apply_chat_template_json(messages_json, true) else {
            self.set_last_error("failed to apply chat template");
            return 0;
        };
        let images = if image_sizes.is_empty() {
            Vec::new()
        } else {
            let Some(images) = copy_image_buffers(images_flat_buffer, image_sizes) else {
                self.set_last_error("media buffers are invalid");
                return 0;
            };
            images
        };
        self.enqueue_prompt_request(
            context_key.to_string(),
            prompt,
            max_tokens,
            images,
            grammar.to_string(),
            token_emission_mode,
        )
    }

    #[cfg(not(target_family = "wasm"))]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start_chat_request(
        &mut self,
        _context_key: &str,
        _messages_json: &str,
        _max_tokens: i32,
        _images_flat_buffer: &[u8],
        _image_sizes: &[i32],
        _token_emission_mode: i32,
        _grammar: &str,
    ) -> u32 {
        0
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn start_embedding_request(
        &mut self,
        context_key: &str,
        input: &str,
        normalize: i32,
    ) -> u32 {
        self.clear_last_error();
        let Some(runtime) = self.inner.runtime.as_mut() else {
            self.set_last_error("runtime is not loaded");
            return 0;
        };
        match runtime.enqueue_embed_request(
            input.to_string(),
            EmbedOptions {
                normalize: normalize != 0,
                context_key: Some(context_key.to_string()),
            },
        ) {
            Ok(request_id) => request_id,
            Err(error) => {
                self.set_last_error(format!("failed to enqueue embedding request: {error:#}"));
                0
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn start_embedding_request(
        &mut self,
        _context_key: &str,
        _input: &str,
        _normalize: i32,
    ) -> u32 {
        0
    }

    #[cfg(target_family = "wasm")]
    fn enqueue_prompt_request(
        &mut self,
        context_key: String,
        prompt: String,
        max_tokens: i32,
        images: Vec<Vec<u8>>,
        grammar: String,
        token_emission_mode: i32,
    ) -> u32 {
        let Some(runtime) = self.inner.runtime.as_mut() else {
            return 0;
        };
        if max_tokens <= 0 {
            return 0;
        }

        let token_emission_mode = emission_mode(token_emission_mode);
        let enqueue_result = if images.is_empty() {
            runtime.enqueue_request(
                context_key,
                prompt,
                max_tokens,
                grammar,
                "",
                Vec::new(),
                None,
                token_emission_mode,
            )
        } else {
            runtime.enqueue_multimodal_request(
                context_key,
                prompt,
                max_tokens,
                images,
                grammar,
                "",
                Vec::new(),
                None,
                token_emission_mode,
            )
        };
        let request_id = match enqueue_result {
            Ok(request_id) => request_id,
            Err(error) => {
                self.set_last_error(format!("failed to enqueue text request: {error:#}"));
                return 0;
            }
        };
        if request_id != 0 {
            if let Some(producer) = self.inner.token_producer.as_ref() {
                runtime
                    .request_queue
                    .token_ring_producers
                    .insert(request_id, producer.clone());
            }
        }
        request_id
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn cancel_request(&mut self, request_id: u32) -> i32 {
        let Some(runtime) = self.inner.runtime.as_mut() else {
            return 0;
        };
        i32::from(runtime.cancel_request(request_id))
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn cancel_request(&mut self, _request_id: u32) -> i32 {
        0
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn run_scheduler_loop(
        &mut self,
        max_ticks: i32,
        max_completed_responses: i32,
        max_emitted_tokens: i32,
        max_duration_us: i32,
        streaming_active: bool,
        out: &mut BrowserSchedulerLoopResult,
    ) -> i32 {
        if self.inner.runtime.is_none() {
            return STATUS_NOT_INITIALIZED;
        }
        let duration = if max_duration_us > 0 {
            Duration::from_micros(max_duration_us as u64)
        } else {
            Duration::ZERO
        };

        // Bulk (non-streaming) path: just run the runtime's own monolithic
        // loop. No per-tick drain/yield overhead — matches the pre-port
        // C++ baseline. The host won't peek at the streaming buffer mid-loop
        // for non-streaming requests anyway, so there's nothing to drain in
        // between.
        if !streaming_active {
            let burst = {
                let Some(runtime) = self.inner.runtime.as_mut() else {
                    return STATUS_NOT_INITIALIZED;
                };
                runtime.run_scheduler_loop(
                    max_ticks,
                    max_completed_responses,
                    max_emitted_tokens,
                    duration,
                )
            };
            self.drain_token_ring();
            *out = scheduler_loop_result_from_runtime(burst);
            return burst.status as i32;
        }

        // Streaming path: yield once per emitted token. Each burst caps at
        // `max_emitted_tokens=1`, so the burst returns as soon as one token
        // is produced (during decode) or step_ticks elapse with no token
        // produced (during prefill). After every burst we drain the token
        // ring into the streaming buffer and call ce_native_yield(), which
        // lets the host write that single token into the SAB ring and post
        // a `streaming-tick` to the main thread — yielding the token-by-token
        // delivery callers expect.
        let deadline = (!duration.is_zero()).then(|| Instant::now() + duration);
        let mut acc = SchedulerBurstResult::default();
        loop {
            let remaining_ticks = if max_ticks > 0 {
                (max_ticks - acc.ticks_executed).max(0)
            } else {
                STREAMING_STEP_TICKS
            };
            if max_ticks > 0 && remaining_ticks == 0 {
                acc.status = RequestStepResult::Progressed;
                break;
            }
            let step_ticks = if max_ticks > 0 {
                STREAMING_STEP_TICKS.min(remaining_ticks.max(1))
            } else {
                STREAMING_STEP_TICKS
            };
            let remaining_completed = if max_completed_responses > 0 {
                (max_completed_responses - acc.completed_response_count).max(0)
            } else {
                0
            };
            // The whole point of the streaming path is "exactly one token per
            // yield". We pass 1 here regardless of the JS-side `max_emitted`
            // budget, which we instead enforce in the outer break check below.
            let step_emitted_budget = 1;
            let step_duration = match deadline {
                Some(deadline) => {
                    let now = Instant::now();
                    if now >= deadline {
                        acc.status = if acc.progressed_ticks > 0 || acc.completed_response_count > 0
                        {
                            RequestStepResult::Progressed
                        } else {
                            RequestStepResult::Waiting
                        };
                        break;
                    }
                    deadline - now
                }
                None => Duration::ZERO,
            };

            let burst = {
                let Some(runtime) = self.inner.runtime.as_mut() else {
                    return STATUS_NOT_INITIALIZED;
                };
                runtime.run_scheduler_burst(
                    step_ticks,
                    remaining_completed,
                    step_emitted_budget,
                    step_duration,
                )
            };
            acc.ticks_executed += burst.ticks_executed;
            acc.progressed_ticks += burst.progressed_ticks;
            acc.completed_response_count += burst.completed_response_count;
            acc.emitted_token_count += burst.emitted_token_count;

            // Drain only when there's something to drain. Prefill bursts that
            // didn't emit a token leave the ring empty; skipping the drain
            // saves a mutex acquisition + Vec setup per prefill burst.
            if burst.emitted_token_count > 0 {
                self.drain_token_ring();
                // SAFETY: This is the Emscripten host boundary declared above;
                // it takes no Rust pointers and drains the already-filled buffer synchronously.
                unsafe {
                    ce_native_yield();
                }
            }

            if matches!(
                burst.status,
                RequestStepResult::Invalid | RequestStepResult::FatalNoProgress
            ) {
                acc.status = burst.status;
                break;
            }
            if burst.status == RequestStepResult::Waiting {
                acc.status = if acc.progressed_ticks > 0 || acc.completed_response_count > 0 {
                    RequestStepResult::Progressed
                } else {
                    RequestStepResult::Waiting
                };
                break;
            }
            if max_completed_responses > 0
                && acc.completed_response_count >= max_completed_responses
            {
                acc.status = RequestStepResult::Progressed;
                break;
            }
            if max_emitted_tokens > 0 && acc.emitted_token_count >= max_emitted_tokens {
                acc.status = RequestStepResult::Progressed;
                break;
            }
        }

        // Tail-drain so any token that arrived just before we hit a limit
        // gets to the SAB ring before the JS scheduler re-enters.
        self.drain_token_ring();
        *out = scheduler_loop_result_from_runtime(acc);
        acc.status as i32
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn run_scheduler_loop(
        &mut self,
        _max_ticks: i32,
        _max_completed_responses: i32,
        _max_emitted_tokens: i32,
        _max_duration_us: i32,
        _streaming_active: bool,
        _out: &mut BrowserSchedulerLoopResult,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    fn drain_token_ring(&mut self) {
        let Some(consumer) = self.inner.token_consumer.as_ref() else {
            return;
        };
        let mut frames = Vec::with_capacity(64);
        loop {
            frames.clear();
            let status = consumer.drain_into(&mut frames, 256, STREAMING_BUFFER_CAPACITY);
            let drop_delta = status
                .drop_count
                .saturating_sub(self.inner.token_ring_drop_count);
            self.inner.token_ring_drop_count = status.drop_count;
            self.inner.streaming_buffer.add_drops(drop_delta);
            if frames.is_empty() {
                break;
            }
            for frame in &frames {
                self.inner
                    .streaming_buffer
                    .try_write_frame(frame.stream_id, &frame.bytes);
            }
            if status.frames_drained == 0 {
                break;
            }
        }
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn completed_status(&self, request_id: u32) -> i32 {
        let Some(runtime) = self.inner.runtime.as_ref() else {
            return COMPLETED_REQUEST_STATUS_UNKNOWN;
        };
        if let Some(response) = runtime.request_queue.completed_responses.get(&request_id) {
            return match response.status {
                GenerateResponseStatus::Pending => COMPLETED_REQUEST_STATUS_PENDING,
                GenerateResponseStatus::Completed => COMPLETED_REQUEST_STATUS_COMPLETED,
                GenerateResponseStatus::Cancelled => COMPLETED_REQUEST_STATUS_CANCELLED,
                GenerateResponseStatus::Failed => COMPLETED_REQUEST_STATUS_FAILED,
            };
        }
        if runtime.request_queue.requests.contains_key(&request_id) {
            COMPLETED_REQUEST_STATUS_PENDING
        } else {
            COMPLETED_REQUEST_STATUS_UNKNOWN
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn completed_status(&self, _request_id: u32) -> i32 {
        COMPLETED_REQUEST_STATUS_UNKNOWN
    }

    #[cfg(target_family = "wasm")]
    fn completed_response_ref(&self, request_id: u32) -> Option<&GenerateResponse> {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.request_queue.completed_responses.get(&request_id))
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn completed_output_kind(&self, request_id: u32) -> i32 {
        self.completed_response_ref(request_id)
            .map(|response| match &response.output {
                ResponseOutput::Text(_) => COMPLETED_REQUEST_OUTPUT_TEXT,
                ResponseOutput::Embedding { .. } => COMPLETED_REQUEST_OUTPUT_EMBEDDING,
            })
            .unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn completed_output_kind(&self, _request_id: u32) -> i32 {
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn completed_embedding_len(&self, request_id: u32) -> i32 {
        self.completed_response_ref(request_id)
            .and_then(|response| match &response.output {
                ResponseOutput::Embedding { values, .. } => value_len_i32(values.len()),
                ResponseOutput::Text(_) => None,
            })
            .unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn completed_embedding_len(&self, _request_id: u32) -> i32 {
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn copy_completed_embedding(&self, request_id: u32, buffer: &mut [f32]) -> i32 {
        let Some(response) = self.completed_response_ref(request_id) else {
            return STATUS_FAILURE;
        };
        let ResponseOutput::Embedding { values, .. } = &response.output else {
            return STATUS_FAILURE;
        };
        if buffer.len() < values.len() {
            return STATUS_INVALID_ARGUMENTS;
        }
        buffer[..values.len()].copy_from_slice(values);
        value_len_i32(values.len()).unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn copy_completed_embedding(&self, _request_id: u32, _buffer: &mut [f32]) -> i32 {
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn completed_embedding_pooling(&self, request_id: u32) -> i32 {
        self.completed_response_ref(request_id)
            .and_then(|response| match &response.output {
                ResponseOutput::Embedding { pooling, .. } => Some(pooling_code(*pooling)),
                ResponseOutput::Text(_) => None,
            })
            .unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn completed_embedding_pooling(&self, _request_id: u32) -> i32 {
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn completed_embedding_normalized(&self, request_id: u32) -> i32 {
        self.completed_response_ref(request_id)
            .and_then(|response| match &response.output {
                ResponseOutput::Embedding { normalized, .. } => Some(i32::from(*normalized)),
                ResponseOutput::Text(_) => None,
            })
            .unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn completed_embedding_normalized(&self, _request_id: u32) -> i32 {
        STATUS_UNAVAILABLE
    }

    pub(crate) fn completed_output_size(&self, request_id: u32) -> i32 {
        completed_output(self, request_id)
            .map(|text| byte_len_i32(text.as_bytes()))
            .unwrap_or(STATUS_FAILURE)
    }

    pub(crate) fn copy_completed_output(&self, request_id: u32, buffer: &mut [u8]) -> i32 {
        completed_output(self, request_id)
            .map(|text| copy_bytes_with_nul(text.as_bytes(), buffer))
            .unwrap_or(STATUS_FAILURE)
    }

    pub(crate) fn completed_error_size(&self, request_id: u32) -> i32 {
        completed_error(self, request_id)
            .map(|text| byte_len_i32(text.as_bytes()))
            .unwrap_or(STATUS_FAILURE)
    }

    pub(crate) fn copy_completed_error(&self, request_id: u32, buffer: &mut [u8]) -> i32 {
        completed_error(self, request_id)
            .map(|text| copy_bytes_with_nul(text.as_bytes(), buffer))
            .unwrap_or(STATUS_FAILURE)
    }

    pub(crate) fn consume_completed_request(&mut self, request_id: u32) -> i32 {
        #[cfg(target_family = "wasm")]
        {
            let Some(runtime) = self.inner.runtime.as_mut() else {
                return 0;
            };
            runtime
                .request_queue
                .token_ring_producers
                .remove(&request_id);
            i32::from(runtime.take_completed_response(request_id).is_some())
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = request_id;
            STATUS_UNAVAILABLE
        }
    }

    pub(crate) fn runtime_observability(&self, out: &mut BrowserRuntimeMetrics) -> i32 {
        #[cfg(target_family = "wasm")]
        {
            let Some(metrics) = self.runtime_metrics() else {
                return STATUS_NOT_INITIALIZED;
            };
            *out = metrics;
            STATUS_OK
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = out;
            STATUS_UNAVAILABLE
        }
    }

    pub(crate) fn completed_runtime_observability(
        &self,
        request_id: u32,
        out: &mut BrowserRuntimeMetrics,
    ) -> i32 {
        #[cfg(target_family = "wasm")]
        {
            let Some(metrics) = self.completed_runtime_metrics(request_id) else {
                return STATUS_FAILURE;
            };
            *out = metrics;
            STATUS_OK
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = request_id;
            let _ = out;
            STATUS_UNAVAILABLE
        }
    }

    #[cfg(target_family = "wasm")]
    fn runtime_metrics(&self) -> Option<BrowserRuntimeMetrics> {
        self.inner
            .runtime
            .as_ref()
            .and_then(InferenceRuntime::try_get_runtime_observability)
            .map(runtime_metrics_from_core)
    }

    #[cfg(target_family = "wasm")]
    fn completed_runtime_metrics(&self, request_id: u32) -> Option<BrowserRuntimeMetrics> {
        self.completed_response_ref(request_id)
            .map(|response| runtime_metrics_from_core(response.runtime_observability))
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn media_marker(&self) -> String {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.media_marker().ok())
            .unwrap_or_default()
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn chat_template_source(&self) -> String {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.chat_template_source().ok().flatten())
            .unwrap_or_default()
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn bos_text(&self) -> String {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.get_bos_text().ok())
            .unwrap_or_default()
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn eos_text(&self) -> String {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.get_eos_text().ok())
            .unwrap_or_default()
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn probe_chat_boundary_info(&self) -> String {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| {
                runtime
                    .probe_chat_boundary_info()
                    .ok()
                    .and_then(|info| serde_json::to_string(&info).ok())
            })
            .unwrap_or_default()
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn streaming_buffer_ptr(&mut self) -> usize {
        self.inner.streaming_buffer.buffer.as_mut_ptr() as usize
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn streaming_buffer_used_address(&mut self) -> usize {
        (&mut self.inner.streaming_buffer.used as *mut i32) as usize
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn streaming_buffer_drop_count_address(&mut self) -> usize {
        (&mut self.inner.streaming_buffer.drop_count as *mut i32) as usize
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn media_marker(&self) -> String {
        String::new()
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn chat_template_source(&self) -> String {
        String::new()
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn bos_text(&self) -> String {
        String::new()
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn eos_text(&self) -> String {
        String::new()
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn probe_chat_boundary_info(&self) -> String {
        String::new()
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn streaming_buffer_ptr(&mut self) -> usize {
        0
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn streaming_buffer_used_address(&mut self) -> usize {
        0
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn streaming_buffer_drop_count_address(&mut self) -> usize {
        0
    }
}

#[cfg(target_family = "wasm")]
impl BrowserEngineInner {
    fn new() -> Self {
        Self {
            runtime: None,
            token_producer: None,
            token_consumer: None,
            token_ring_drop_count: 0,
            streaming_buffer: StreamingBuffer::new(STREAMING_BUFFER_CAPACITY),
        }
    }
}

#[cfg(not(target_family = "wasm"))]
impl BrowserEngineInner {
    fn new() -> Self {
        Self
    }
}

#[cfg(target_family = "wasm")]
struct StreamingBuffer {
    buffer: Vec<u8>,
    used: i32,
    drop_count: i32,
}

#[cfg(target_family = "wasm")]
impl StreamingBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0; capacity],
            used: 0,
            drop_count: 0,
        }
    }

    fn reset(&mut self) {
        self.used = 0;
        self.drop_count = 0;
    }

    fn add_drops(&mut self, count: u64) {
        if count > 0 {
            self.drop_count = self
                .drop_count
                .saturating_add(count.min(i32::MAX as u64) as i32);
        }
    }

    fn try_write_frame(&mut self, request_id: u32, bytes: &[u8]) -> bool {
        if request_id == 0 || bytes.is_empty() {
            return true;
        }
        let used = self.used.max(0) as usize;
        let record_len = STREAMING_RECORD_HEADER_BYTES + bytes.len();
        if record_len > self.buffer.len() || used.saturating_add(record_len) > self.buffer.len() {
            self.drop_count = self.drop_count.saturating_add(1);
            return false;
        }
        let offset = used;
        self.buffer[offset..offset + 4].copy_from_slice(&request_id.to_le_bytes());
        self.buffer[offset + 4..offset + 8].copy_from_slice(&(bytes.len() as u32).to_le_bytes());
        self.buffer[offset + 8..offset + record_len].copy_from_slice(bytes);
        self.used = (used + record_len) as i32;
        true
    }
}

fn byte_len_i32(bytes: &[u8]) -> i32 {
    i32::try_from(bytes.len()).unwrap_or(STATUS_FAILURE)
}

#[cfg(target_family = "wasm")]
fn value_len_i32(len: usize) -> Option<i32> {
    i32::try_from(len).ok()
}

fn copy_bytes_with_nul(bytes: &[u8], buffer: &mut [u8]) -> i32 {
    if buffer.len() <= bytes.len() {
        return STATUS_INVALID_ARGUMENTS;
    }
    buffer[..bytes.len()].copy_from_slice(bytes);
    buffer[bytes.len()] = 0;
    byte_len_i32(bytes)
}

#[cfg(target_family = "wasm")]
fn read_runtime_config(raw: &str) -> Result<NativeRuntimeConfig, String> {
    let json = if raw.trim().is_empty() {
        "{}"
    } else {
        raw.trim()
    };
    serde_json::from_str::<NativeRuntimeConfig>(json)
        .map_err(|error| format!("runtime config JSON is invalid: {error}"))
}

#[cfg(target_family = "wasm")]
fn emission_mode(token_emission_mode: i32) -> GenerateTokenEmissionMode {
    if token_emission_mode == 1 {
        GenerateTokenEmissionMode::TokenStream
    } else {
        GenerateTokenEmissionMode::None
    }
}

#[cfg(target_family = "wasm")]
fn pooling_code(pooling: PoolingType) -> i32 {
    match pooling {
        PoolingType::Unspecified => -1,
        PoolingType::None => 0,
        PoolingType::Mean => 1,
        PoolingType::Cls => 2,
        PoolingType::Last => 3,
        PoolingType::Rank => 4,
    }
}

#[cfg(target_family = "wasm")]
fn copy_image_buffers(images_flat_buffer: &[u8], image_sizes: &[i32]) -> Option<Vec<Vec<u8>>> {
    let total_bytes = image_sizes.iter().try_fold(0usize, |sum, size| {
        let size = usize::try_from(*size).ok()?;
        sum.checked_add(size)
    })?;
    let flat = images_flat_buffer.get(..total_bytes)?;
    let mut images = Vec::with_capacity(image_sizes.len());
    let mut offset = 0usize;
    for size in image_sizes {
        let size = usize::try_from(*size).ok()?;
        let end = offset.checked_add(size)?;
        images.push(flat.get(offset..end)?.to_vec());
        offset = end;
    }
    Some(images)
}

#[cfg(target_family = "wasm")]
fn scheduler_loop_result_from_runtime(result: SchedulerBurstResult) -> BrowserSchedulerLoopResult {
    BrowserSchedulerLoopResult {
        ticks_executed: result.ticks_executed,
        progressed_ticks: result.progressed_ticks,
        completed_response_count: result.completed_response_count,
        emitted_token_count: result.emitted_token_count,
    }
}

#[cfg(target_family = "wasm")]
fn runtime_metrics_from_core(
    metrics: cogentlm_engine::runtime::metrics::RuntimeObservabilityMetrics,
) -> BrowserRuntimeMetrics {
    BrowserRuntimeMetrics {
        ttft_ms: metrics.ttft_ms,
        itl_avg_ms: metrics.itl_avg_ms,
        itl_p99_ms: metrics.itl_p99_ms,
        e2e_ms: metrics.e2e_ms,
        prefill_ms: metrics.prefill_ms,
        decode_ms: metrics.decode_ms,
        native_gpu_ms: metrics.native_gpu_ms,
        native_sync_ms: metrics.native_sync_ms,
        native_logic_ms: metrics.native_logic_ms,
        input_tokens: metrics.input_tokens,
        output_tokens: metrics.output_tokens,
        cache_hits: metrics.cache_hits,
        prefill_tokens: metrics.prefill_tokens,
    }
}

#[cfg(target_family = "wasm")]
fn completed_output(engine: &BrowserEngine, request_id: u32) -> Option<&str> {
    engine
        .completed_response_ref(request_id)
        .and_then(|response| match &response.output {
            ResponseOutput::Text(text) => Some(text.as_str()),
            ResponseOutput::Embedding { .. } => None,
        })
}

#[cfg(not(target_family = "wasm"))]
fn completed_output(_engine: &BrowserEngine, _request_id: u32) -> Option<&str> {
    None
}

#[cfg(target_family = "wasm")]
fn completed_error(engine: &BrowserEngine, request_id: u32) -> Option<&str> {
    engine
        .completed_response_ref(request_id)
        .map(|response| response.error_message.as_str())
}

#[cfg(not(target_family = "wasm"))]
fn completed_error(_engine: &BrowserEngine, _request_id: u32) -> Option<&str> {
    None
}

#[cfg(test)]
#[path = "tests/root_tests.rs"]
mod root_tests;
