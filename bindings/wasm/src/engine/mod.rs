use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(target_family = "wasm")]
use std::cell::UnsafeCell;
#[cfg(target_family = "wasm")]
use std::collections::HashMap;
#[cfg(target_family = "wasm")]
use std::fmt;
#[cfg(target_family = "wasm")]
use std::sync::{Arc, Mutex};
#[cfg(target_family = "wasm")]
use std::time::Duration;

#[cfg(target_family = "wasm")]
use sipp::backend::set_llama_log_quiet;
#[cfg(target_family = "wasm")]
use sipp::engine::protocol::{EmbedOptions, PoolingType};
#[cfg(target_family = "wasm")]
use sipp::runtime::config::{NativeRuntimeConfig, RequestSampling, SamplingRuntimePatch};
#[cfg(target_family = "wasm")]
use sipp::runtime::request::{
    GenerateResponse, GenerateResponseStatus, ResponseOutput, TokenEmissionSink,
    TokenEmissionSinkRef,
};
#[cfg(target_family = "wasm")]
use sipp::runtime::{InferenceRuntime, SchedulerBurstResult};

use crate::{BrowserRuntimeMetrics, BrowserSchedulerLoopResult};

pub(crate) const ABI_VERSION: u32 = 6;

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
const SHARED_TOKEN_RING_CAPACITY: usize = 256 * 1024;
#[cfg(target_family = "wasm")]
const SHARED_TOKEN_RING_RECORD_HEADER_BYTES: usize = 16;

static NEXT_ENGINE_ID: AtomicU32 = AtomicU32::new(1);

#[repr(C)]
pub struct BrowserEngine {
    id: u32,
    last_error: String,
    inner: BrowserEngineInner,
}

pub(crate) struct BrowserTextRequestArgs<'a> {
    pub(crate) emit_tokens: i32,
    pub(crate) grammar: &'a str,
    pub(crate) stop_json: &'a str,
    pub(crate) sampling_json: &'a str,
}

pub(crate) struct BrowserMediaInput<'a> {
    pub(crate) flat_buffer: &'a [u8],
    pub(crate) sizes: &'a [i32],
}

#[cfg(target_family = "wasm")]
struct BrowserPromptRequest {
    context_key: String,
    prompt: String,
    max_tokens: i32,
    images: Vec<Vec<u8>>,
    grammar: String,
    stop: Vec<String>,
    sampling: Option<RequestSampling>,
    emit_tokens: i32,
}

#[cfg(target_family = "wasm")]
struct BrowserEngineInner {
    runtime: Option<InferenceRuntime>,
    token_ring: SharedTokenRing,
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
        self.inner.token_ring.reset();
        self.inner.runtime = Some(runtime);
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
        self.inner.token_ring.reset();
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn start_text_request(
        &mut self,
        context_key: &str,
        prompt: &str,
        max_tokens: i32,
        args: BrowserTextRequestArgs<'_>,
    ) -> u32 {
        self.clear_last_error();
        let Some((stop, sampling)) =
            self.parse_text_request_options(args.stop_json, args.sampling_json)
        else {
            return 0;
        };
        self.enqueue_prompt_request(BrowserPromptRequest {
            context_key: context_key.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            images: Vec::new(),
            grammar: args.grammar.to_string(),
            stop,
            sampling,
            emit_tokens: args.emit_tokens,
        })
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn start_text_request(
        &mut self,
        _context_key: &str,
        _prompt: &str,
        _max_tokens: i32,
        _args: BrowserTextRequestArgs<'_>,
    ) -> u32 {
        0
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn start_media_request(
        &mut self,
        context_key: &str,
        prompt: &str,
        max_tokens: i32,
        media: BrowserMediaInput<'_>,
        args: BrowserTextRequestArgs<'_>,
    ) -> u32 {
        self.clear_last_error();
        let Some(images) = copy_image_buffers(media.flat_buffer, media.sizes) else {
            self.set_last_error("media buffers are invalid");
            return 0;
        };
        let Some((stop, sampling)) =
            self.parse_text_request_options(args.stop_json, args.sampling_json)
        else {
            return 0;
        };

        self.enqueue_prompt_request(BrowserPromptRequest {
            context_key: context_key.to_string(),
            prompt: prompt.to_string(),
            max_tokens,
            images,
            grammar: args.grammar.to_string(),
            stop,
            sampling,
            emit_tokens: args.emit_tokens,
        })
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn start_media_request(
        &mut self,
        _context_key: &str,
        _prompt: &str,
        _max_tokens: i32,
        _media: BrowserMediaInput<'_>,
        _args: BrowserTextRequestArgs<'_>,
    ) -> u32 {
        0
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn start_chat_request(
        &mut self,
        context_key: &str,
        messages_json: &str,
        max_tokens: i32,
        media: BrowserMediaInput<'_>,
        args: BrowserTextRequestArgs<'_>,
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
        let images = if media.sizes.is_empty() {
            Vec::new()
        } else {
            let Some(images) = copy_image_buffers(media.flat_buffer, media.sizes) else {
                self.set_last_error("media buffers are invalid");
                return 0;
            };
            images
        };
        let Some((stop, sampling)) =
            self.parse_text_request_options(args.stop_json, args.sampling_json)
        else {
            return 0;
        };
        self.enqueue_prompt_request(BrowserPromptRequest {
            context_key: context_key.to_string(),
            prompt,
            max_tokens,
            images,
            grammar: args.grammar.to_string(),
            stop,
            sampling,
            emit_tokens: args.emit_tokens,
        })
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn start_chat_request(
        &mut self,
        _context_key: &str,
        _messages_json: &str,
        _max_tokens: i32,
        _media: BrowserMediaInput<'_>,
        _args: BrowserTextRequestArgs<'_>,
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
    fn parse_text_request_options(
        &mut self,
        stop_json: &str,
        sampling_json: &str,
    ) -> Option<(Vec<String>, Option<RequestSampling>)> {
        let stop = if stop_json.is_empty() {
            Vec::new()
        } else {
            match serde_json::from_str::<Vec<String>>(stop_json) {
                Ok(stop) => stop,
                Err(error) => {
                    self.set_last_error(format!("invalid stop JSON: {error}"));
                    return None;
                }
            }
        };
        let sampling = if sampling_json.is_empty() {
            None
        } else {
            match serde_json::from_str::<SamplingRuntimePatch>(sampling_json) {
                Ok(patch) if patch.temperature.is_some() || patch.top_p.is_some() => {
                    Some(RequestSampling::Patch(patch))
                }
                Ok(_) => None,
                Err(error) => {
                    self.set_last_error(format!("invalid sampling JSON: {error}"));
                    return None;
                }
            }
        };
        Some((stop, sampling))
    }

    #[cfg(target_family = "wasm")]
    fn enqueue_prompt_request(&mut self, request: BrowserPromptRequest) -> u32 {
        let Some(runtime) = self.inner.runtime.as_mut() else {
            return 0;
        };
        if request.max_tokens <= 0 {
            return 0;
        }

        let emit_tokens = emit_tokens_enabled(request.emit_tokens);
        let enqueue_result = if request.images.is_empty() {
            runtime.enqueue_request(
                request.context_key,
                request.prompt,
                request.max_tokens,
                request.grammar,
                "",
                request.stop,
                request.sampling,
                emit_tokens,
            )
        } else {
            runtime.enqueue_multimodal_request(
                request.context_key,
                request.prompt,
                request.max_tokens,
                request.images,
                request.grammar,
                "",
                request.stop,
                request.sampling,
                emit_tokens,
            )
        };
        let request_id = match enqueue_result {
            Ok(request_id) => request_id,
            Err(error) => {
                self.set_last_error(format!("failed to enqueue text request: {error:#}"));
                return 0;
            }
        };
        if emit_tokens && request_id != 0 {
            runtime
                .request_queue
                .token_emission_sinks
                .insert(request_id, self.inner.token_ring.sink());
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
        max_generated_tokens: i32,
        max_duration_us: i32,
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

        let Some(runtime) = self.inner.runtime.as_mut() else {
            return STATUS_NOT_INITIALIZED;
        };
        let burst = runtime.run_scheduler_loop(
            max_ticks,
            max_completed_responses,
            max_generated_tokens,
            duration,
        );
        *out = scheduler_loop_result_from_runtime(burst);
        burst.status as i32
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn run_scheduler_loop(
        &mut self,
        _max_ticks: i32,
        _max_completed_responses: i32,
        _max_generated_tokens: i32,
        _max_duration_us: i32,
        _out: &mut BrowserSchedulerLoopResult,
    ) -> i32 {
        STATUS_UNAVAILABLE
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
        if runtime.request_queue.contains_request(request_id) {
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
                .token_emission_sinks
                .remove(&request_id);
            self.inner.token_ring.forget_stream(request_id);
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
    pub(crate) fn token_ring_header_address(&self) -> usize {
        self.inner.token_ring.header_address() as usize
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn token_ring_body_address(&self) -> usize {
        self.inner.token_ring.body_address() as usize
    }

    #[cfg(target_family = "wasm")]
    pub(crate) fn token_ring_capacity(&self) -> i32 {
        self.inner.token_ring.capacity_i32()
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
    pub(crate) fn token_ring_header_address(&self) -> usize {
        0
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn token_ring_body_address(&self) -> usize {
        0
    }

    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn token_ring_capacity(&self) -> i32 {
        0
    }
}

#[cfg(target_family = "wasm")]
impl BrowserEngineInner {
    fn new() -> Self {
        Self {
            runtime: None,
            token_ring: SharedTokenRing::new(SHARED_TOKEN_RING_CAPACITY),
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
struct SharedTokenRing {
    inner: Arc<SharedTokenRingInner>,
}

#[cfg(target_family = "wasm")]
#[derive(Debug)]
struct SharedTokenRingInner {
    header: Box<SharedTokenRingHeader>,
    body: SharedTokenRingBody,
    sequences: Mutex<HashMap<u32, u32>>,
}

#[cfg(target_family = "wasm")]
struct SharedTokenRingBody {
    bytes: UnsafeCell<Box<[u8]>>,
}

#[cfg(target_family = "wasm")]
#[repr(C, align(4))]
#[derive(Debug)]
struct SharedTokenRingHeader {
    write_index: AtomicU32,
    read_index: AtomicU32,
    capacity: AtomicU32,
    drop_count: AtomicU32,
    reserved: [AtomicU32; 4],
}

#[cfg(target_family = "wasm")]
#[derive(Debug)]
struct SharedTokenRingSink {
    inner: Arc<SharedTokenRingInner>,
}

#[cfg(target_family = "wasm")]
impl SharedTokenRing {
    fn new(capacity: usize) -> Self {
        let inner = Arc::new(SharedTokenRingInner {
            header: Box::new(SharedTokenRingHeader {
                write_index: AtomicU32::new(0),
                read_index: AtomicU32::new(0),
                capacity: AtomicU32::new(capacity as u32),
                drop_count: AtomicU32::new(0),
                reserved: [
                    AtomicU32::new(0),
                    AtomicU32::new(0),
                    AtomicU32::new(0),
                    AtomicU32::new(0),
                ],
            }),
            body: SharedTokenRingBody::new(capacity),
            sequences: Mutex::new(HashMap::new()),
        });
        Self { inner }
    }

    fn reset(&mut self) {
        self.inner.header.write_index.store(0, Ordering::Release);
        self.inner.header.read_index.store(0, Ordering::Release);
        self.inner.header.drop_count.store(0, Ordering::Release);
        lock_sequences(&self.inner.sequences).clear();
    }

    fn sink(&self) -> TokenEmissionSinkRef {
        Arc::new(SharedTokenRingSink {
            inner: Arc::clone(&self.inner),
        })
    }

    fn header_address(&self) -> *const u32 {
        (&*self.inner.header as *const SharedTokenRingHeader).cast()
    }

    fn body_address(&self) -> *const u8 {
        self.inner.body.as_ptr()
    }

    fn capacity_i32(&self) -> i32 {
        i32::try_from(self.inner.body.len()).unwrap_or(i32::MAX)
    }

    fn forget_stream(&self, stream_id: u32) {
        lock_sequences(&self.inner.sequences).remove(&stream_id);
    }
}

#[cfg(target_family = "wasm")]
impl TokenEmissionSink for SharedTokenRingSink {
    fn try_write_batch(&self, stream_id: u32, frame_count: u32, bytes: &[u8]) -> bool {
        if stream_id == 0 || frame_count == 0 || bytes.is_empty() {
            return true;
        }
        let record_size = SHARED_TOKEN_RING_RECORD_HEADER_BYTES.saturating_add(bytes.len());
        let capacity = self.inner.body.len();
        if record_size > capacity {
            self.inner.header.drop_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let write_index = self.inner.header.write_index.load(Ordering::Acquire);
        let read_index = self.inner.header.read_index.load(Ordering::Acquire);
        let used = write_index.wrapping_sub(read_index) as usize;
        if used.saturating_add(record_size) > capacity {
            self.inner.header.drop_count.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        let sequence_start = next_sequence(&self.inner.sequences, stream_id, frame_count);
        let offset = (write_index as usize) % capacity;
        self.inner.body.with_mut(|body| {
            write_wrapped_u32(body, offset, stream_id);
            write_wrapped_u32(body, offset + 4, sequence_start);
            write_wrapped_u32(body, offset + 8, frame_count);
            write_wrapped_u32(body, offset + 12, bytes.len() as u32);
            write_wrapped_bytes(body, offset + SHARED_TOKEN_RING_RECORD_HEADER_BYTES, bytes);
        });
        self.inner.header.write_index.store(
            write_index.wrapping_add(record_size as u32),
            Ordering::Release,
        );
        true
    }

    fn close(&self) {}
}

#[cfg(target_family = "wasm")]
fn next_sequence(sequences: &Mutex<HashMap<u32, u32>>, stream_id: u32, frame_count: u32) -> u32 {
    let mut sequences = lock_sequences(sequences);
    let sequence = sequences.get(&stream_id).copied().unwrap_or(0);
    sequences.insert(stream_id, sequence.wrapping_add(frame_count));
    sequence
}

#[cfg(target_family = "wasm")]
fn lock_sequences(
    sequences: &Mutex<HashMap<u32, u32>>,
) -> std::sync::MutexGuard<'_, HashMap<u32, u32>> {
    match sequences.lock() {
        Ok(sequences) => sequences,
        Err(error) => error.into_inner(),
    }
}

#[cfg(target_family = "wasm")]
impl SharedTokenRingBody {
    fn new(capacity: usize) -> Self {
        Self {
            bytes: UnsafeCell::new(vec![0; capacity].into_boxed_slice()),
        }
    }

    fn as_ptr(&self) -> *const u8 {
        // SAFETY: the boxed slice is allocated once and never reallocated or
        // moved out of the cell after construction.
        unsafe { (&*self.bytes.get()).as_ptr() }
    }

    fn len(&self) -> usize {
        // SAFETY: reading the slice length does not touch mutable bytes.
        unsafe { (&*self.bytes.get()).len() }
    }

    fn with_mut(&self, write: impl FnOnce(&mut [u8])) {
        // SAFETY: the native runtime is the only writer. JS reads records only
        // after the producer publishes write_index with Release ordering; the
        // JS reader observes that index with Atomics.load before reading bytes.
        unsafe { write((&mut *self.bytes.get()).as_mut()) }
    }
}

#[cfg(target_family = "wasm")]
// SAFETY: `SharedTokenRingBody` owns a stable byte buffer. The native side has
// one producer, and cross-thread visibility is controlled by atomic ring
// indices; JS readers do not obtain Rust references.
unsafe impl Send for SharedTokenRingBody {}

#[cfg(target_family = "wasm")]
// SAFETY: shared access only exposes raw wasm memory to JS. Native mutation
// happens through the single producer before the Release store to write_index.
unsafe impl Sync for SharedTokenRingBody {}

#[cfg(target_family = "wasm")]
impl fmt::Debug for SharedTokenRingBody {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SharedTokenRingBody")
            .field("len", &self.len())
            .finish()
    }
}

#[cfg(target_family = "wasm")]
fn write_wrapped_u32(body: &mut [u8], offset: usize, value: u32) {
    let bytes = value.to_le_bytes();
    write_wrapped_bytes(body, offset, &bytes);
}

#[cfg(target_family = "wasm")]
fn write_wrapped_bytes(body: &mut [u8], offset: usize, bytes: &[u8]) {
    let capacity = body.len();
    let offset = offset % capacity;
    let tail = capacity - offset;
    if bytes.len() <= tail {
        body[offset..offset + bytes.len()].copy_from_slice(bytes);
        return;
    }
    body[offset..].copy_from_slice(&bytes[..tail]);
    body[..bytes.len() - tail].copy_from_slice(&bytes[tail..]);
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
fn emit_tokens_enabled(emit_tokens: i32) -> bool {
    emit_tokens != 0
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
    metrics: sipp::runtime::metrics::RuntimeObservabilityMetrics,
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
        cache_mode: metrics.cache_mode as i32,
        cache_source: metrics.cache_source as i32,
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
#[path = "../tests/engine/root_tests.rs"]
mod root_tests;
