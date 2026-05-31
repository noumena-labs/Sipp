use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(target_family = "wasm")]
use std::time::Duration;

#[cfg(target_family = "wasm")]
use cogentlm_engine::backend::{backend_observability_json, set_llama_log_quiet};
#[cfg(target_family = "wasm")]
use cogentlm_engine::engine::protocol::{EmbedOptions, PoolingType};
#[cfg(target_family = "wasm")]
use cogentlm_engine::runtime::config::NativeRuntimeConfig;
#[cfg(target_family = "wasm")]
use cogentlm_engine::runtime::request::{
    token_byte_ring, GenerateResponse, GenerateResponseStatus, ResponseOutput,
    TokenByteRingConsumer, TokenByteRingProducer, TokenRingFrame, TOKEN_RING_DEFAULT_CAPACITY,
};
#[cfg(target_family = "wasm")]
use cogentlm_engine::runtime::{InferenceRuntime, SchedulerBurstResult};

use crate::ffi::free_c_string;
#[cfg(target_family = "wasm")]
use crate::ffi::{into_c_string, read_optional_c_string};

const ABI_VERSION: u32 = 5;

const STATUS_OK: i32 = 0;
const STATUS_FAILURE: i32 = -1;
const STATUS_INVALID_ARGUMENTS: i32 = -2;
const STATUS_PANIC: i32 = -3;
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
const TOKEN_BUFFER_CAPACITY: usize = 256 * 1024;
#[cfg(target_family = "wasm")]
const TOKEN_RECORD_HEADER_BYTES: usize = 16;

static NEXT_ENGINE_ID: AtomicU32 = AtomicU32::new(1);

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BrowserSchedulerLoopResult {
    pub ticks_executed: i32,
    pub progressed_ticks: i32,
    pub completed_response_count: i32,
    pub emitted_token_count: i32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BrowserRuntimeMetrics {
    pub ttft_ms: f64,
    pub itl_avg_ms: f64,
    pub itl_p99_ms: f64,
    pub e2e_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    pub native_gpu_ms: f64,
    pub native_sync_ms: f64,
    pub native_logic_ms: f64,
    pub input_tokens: i32,
    pub output_tokens: i32,
    pub cache_hits: i32,
    pub prefill_tokens: i32,
}

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
    token_buffer: TokenBuffer,
}

#[cfg(not(target_family = "wasm"))]
struct BrowserEngineInner;

impl BrowserEngine {
    fn new(id: u32) -> Self {
        Self {
            id,
            last_error: String::new(),
            inner: BrowserEngineInner::new(),
        }
    }

    #[cfg(target_family = "wasm")]
    fn load(&mut self, model_path: *const c_char, runtime_config_json: *const c_char) -> i32 {
        self.clear_last_error();

        let Some(model_path) = read_c_string(model_path) else {
            self.set_last_error("model path is missing or is not valid UTF-8");
            return STATUS_INVALID_ARGUMENTS;
        };
        let runtime_config = match read_runtime_config(runtime_config_json) {
            Ok(config) => config,
            Err(error) => {
                self.set_last_error(error);
                return STATUS_INVALID_ARGUMENTS;
            }
        };

        if model_path.trim().is_empty() {
            self.set_last_error("model path is empty");
            return STATUS_INVALID_ARGUMENTS;
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
    fn load(&mut self, _model_path: *const c_char, _runtime_config_json: *const c_char) -> i32 {
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

    fn last_error_size(&self) -> i32 {
        byte_len_i32(self.last_error.as_bytes())
    }

    fn copy_last_error(&self, buffer: *mut u8, buffer_len: usize) -> i32 {
        copy_bytes_with_nul(self.last_error.as_bytes(), buffer, buffer_len)
    }

    #[cfg(target_family = "wasm")]
    fn close_runtime(&mut self) {
        self.inner.runtime = None;
        self.inner.token_producer = None;
        self.inner.token_consumer = None;
        self.inner.token_buffer.reset();
    }

    #[cfg(target_family = "wasm")]
    fn start_text_request(
        &mut self,
        context_key: *const c_char,
        prompt: *const c_char,
        max_tokens: i32,
        emit_tokens: i32,
        grammar: *const c_char,
    ) -> u32 {
        self.clear_last_error();
        let Some(prompt) = read_c_string(prompt) else {
            self.set_last_error("prompt is missing or is not valid UTF-8");
            return 0;
        };

        let context_key = read_optional_c_string(context_key).unwrap_or_default();
        let grammar = read_optional_c_string(grammar).unwrap_or_default();
        self.enqueue_prompt_request(
            context_key,
            prompt,
            max_tokens,
            Vec::new(),
            grammar,
            emit_tokens,
        )
    }

    #[cfg(not(target_family = "wasm"))]
    fn start_text_request(
        &mut self,
        _context_key: *const c_char,
        _prompt: *const c_char,
        _max_tokens: i32,
        _emit_tokens: i32,
        _grammar: *const c_char,
    ) -> u32 {
        0
    }

    #[cfg(target_family = "wasm")]
    fn start_media_request(
        &mut self,
        context_key: *const c_char,
        prompt: *const c_char,
        max_tokens: i32,
        image_count: i32,
        images_flat_buffer: *const u8,
        image_sizes: *const i32,
        emit_tokens: i32,
        grammar: *const c_char,
    ) -> u32 {
        self.clear_last_error();
        let Some(prompt) = read_c_string(prompt) else {
            self.set_last_error("prompt is missing or is not valid UTF-8");
            return 0;
        };
        let Some(images) = copy_image_buffers(image_count, images_flat_buffer, image_sizes) else {
            self.set_last_error("media buffers are invalid");
            return 0;
        };

        let context_key = read_optional_c_string(context_key).unwrap_or_default();
        let grammar = read_optional_c_string(grammar).unwrap_or_default();
        self.enqueue_prompt_request(
            context_key,
            prompt,
            max_tokens,
            images,
            grammar,
            emit_tokens,
        )
    }

    #[cfg(not(target_family = "wasm"))]
    #[allow(clippy::too_many_arguments)]
    fn start_media_request(
        &mut self,
        _context_key: *const c_char,
        _prompt: *const c_char,
        _max_tokens: i32,
        _image_count: i32,
        _images_flat_buffer: *const u8,
        _image_sizes: *const i32,
        _emit_tokens: i32,
        _grammar: *const c_char,
    ) -> u32 {
        0
    }

    #[cfg(target_family = "wasm")]
    #[allow(clippy::too_many_arguments)]
    fn start_chat_request(
        &mut self,
        context_key: *const c_char,
        messages_json: *const c_char,
        max_tokens: i32,
        image_count: i32,
        images_flat_buffer: *const u8,
        image_sizes: *const i32,
        emit_tokens: i32,
        grammar: *const c_char,
    ) -> u32 {
        self.clear_last_error();
        let Some(runtime) = self.inner.runtime.as_ref() else {
            self.set_last_error("runtime is not loaded");
            return 0;
        };
        let Some(messages_json) = read_c_string(messages_json) else {
            self.set_last_error("messages JSON is missing or is not valid UTF-8");
            return 0;
        };
        let Ok(prompt) = runtime.apply_chat_template_json(&messages_json, true) else {
            self.set_last_error("failed to apply chat template");
            return 0;
        };
        if prompt.is_empty() {
            self.set_last_error("chat template produced an empty prompt");
            return 0;
        }
        let images = if image_count > 0 {
            let Some(images) = copy_image_buffers(image_count, images_flat_buffer, image_sizes)
            else {
                self.set_last_error("media buffers are invalid");
                return 0;
            };
            images
        } else {
            Vec::new()
        };
        let context_key = read_optional_c_string(context_key).unwrap_or_default();
        let grammar = read_optional_c_string(grammar).unwrap_or_default();
        self.enqueue_prompt_request(
            context_key,
            prompt,
            max_tokens,
            images,
            grammar,
            emit_tokens,
        )
    }

    #[cfg(not(target_family = "wasm"))]
    #[allow(clippy::too_many_arguments)]
    fn start_chat_request(
        &mut self,
        _context_key: *const c_char,
        _messages_json: *const c_char,
        _max_tokens: i32,
        _image_count: i32,
        _images_flat_buffer: *const u8,
        _image_sizes: *const i32,
        _emit_tokens: i32,
        _grammar: *const c_char,
    ) -> u32 {
        0
    }

    #[cfg(target_family = "wasm")]
    fn start_embedding_request(
        &mut self,
        context_key: *const c_char,
        input: *const c_char,
        normalize: i32,
    ) -> u32 {
        self.clear_last_error();
        let Some(input) = read_c_string(input) else {
            self.set_last_error("embedding input is missing or is not valid UTF-8");
            return 0;
        };
        let context_key = read_optional_c_string(context_key).unwrap_or_default();
        let Some(runtime) = self.inner.runtime.as_mut() else {
            self.set_last_error("runtime is not loaded");
            return 0;
        };
        match runtime.enqueue_embed_request(
            input,
            EmbedOptions {
                normalize: normalize != 0,
                context_key: Some(context_key),
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
    fn start_embedding_request(
        &mut self,
        _context_key: *const c_char,
        _input: *const c_char,
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
        emit_tokens: i32,
    ) -> u32 {
        let Some(runtime) = self.inner.runtime.as_mut() else {
            return 0;
        };
        if max_tokens <= 0 {
            return 0;
        }

        let emit_tokens = emit_tokens_enabled(emit_tokens);
        let enqueue_result = if images.is_empty() {
            runtime.enqueue_request(
                context_key,
                prompt,
                max_tokens,
                grammar,
                "",
                Vec::new(),
                None,
                emit_tokens,
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
    fn cancel_request(&mut self, request_id: u32) -> i32 {
        let Some(runtime) = self.inner.runtime.as_mut() else {
            return 0;
        };
        i32::from(runtime.cancel_request(request_id))
    }

    #[cfg(not(target_family = "wasm"))]
    fn cancel_request(&mut self, _request_id: u32) -> i32 {
        0
    }

    #[cfg(target_family = "wasm")]
    fn run_scheduler_loop(
        &mut self,
        max_ticks: i32,
        max_completed_responses: i32,
        max_generated_tokens: i32,
        max_duration_us: i32,
        out: *mut BrowserSchedulerLoopResult,
    ) -> i32 {
        if out.is_null() {
            return STATUS_INVALID_ARGUMENTS;
        }
        if self.inner.runtime.is_none() {
            return STATUS_NOT_INITIALIZED;
        }
        let duration = if max_duration_us > 0 {
            Duration::from_micros(max_duration_us as u64)
        } else {
            Duration::ZERO
        };

        let runtime = self
            .inner
            .runtime
            .as_mut()
            .expect("runtime present after early return");
        let burst = runtime.run_scheduler_loop(
            max_ticks,
            max_completed_responses,
            max_generated_tokens,
            duration,
        );
        self.drain_token_ring();
        unsafe {
            *out = scheduler_loop_result_from_runtime(burst);
        }
        burst.status as i32
    }

    #[cfg(not(target_family = "wasm"))]
    fn run_scheduler_loop(
        &mut self,
        _max_ticks: i32,
        _max_completed_responses: i32,
        _max_generated_tokens: i32,
        _max_duration_us: i32,
        _out: *mut BrowserSchedulerLoopResult,
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
            let status = consumer.drain_into(&mut frames, 256, TOKEN_BUFFER_CAPACITY);
            if frames.is_empty() {
                break;
            }
            self.inner.token_buffer.write_frames(&frames);
            if status.frames_drained == 0 {
                break;
            }
        }
    }

    #[cfg(target_family = "wasm")]
    fn completed_status(&self, request_id: u32) -> i32 {
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
    fn completed_status(&self, _request_id: u32) -> i32 {
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
    fn completed_output_kind(&self, request_id: u32) -> i32 {
        self.completed_response_ref(request_id)
            .map(|response| match &response.output {
                ResponseOutput::Text(_) => COMPLETED_REQUEST_OUTPUT_TEXT,
                ResponseOutput::Embedding { .. } => COMPLETED_REQUEST_OUTPUT_EMBEDDING,
            })
            .unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    fn completed_output_kind(&self, _request_id: u32) -> i32 {
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    fn completed_embedding_len(&self, request_id: u32) -> i32 {
        self.completed_response_ref(request_id)
            .and_then(|response| match &response.output {
                ResponseOutput::Embedding { values, .. } => value_len_i32(values.len()),
                ResponseOutput::Text(_) => None,
            })
            .unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    fn completed_embedding_len(&self, _request_id: u32) -> i32 {
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    fn copy_completed_embedding(
        &self,
        request_id: u32,
        buffer: *mut f32,
        value_count: usize,
    ) -> i32 {
        if buffer.is_null() {
            return STATUS_INVALID_ARGUMENTS;
        }
        let Some(response) = self.completed_response_ref(request_id) else {
            return STATUS_FAILURE;
        };
        let ResponseOutput::Embedding { values, .. } = &response.output else {
            return STATUS_FAILURE;
        };
        if value_count < values.len() {
            return STATUS_INVALID_ARGUMENTS;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), buffer, values.len());
        }
        value_len_i32(values.len()).unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    fn copy_completed_embedding(
        &self,
        _request_id: u32,
        _buffer: *mut f32,
        _value_count: usize,
    ) -> i32 {
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    fn completed_embedding_pooling(&self, request_id: u32) -> i32 {
        self.completed_response_ref(request_id)
            .and_then(|response| match &response.output {
                ResponseOutput::Embedding { pooling, .. } => Some(pooling_code(*pooling)),
                ResponseOutput::Text(_) => None,
            })
            .unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    fn completed_embedding_pooling(&self, _request_id: u32) -> i32 {
        STATUS_UNAVAILABLE
    }

    #[cfg(target_family = "wasm")]
    fn completed_embedding_normalized(&self, request_id: u32) -> i32 {
        self.completed_response_ref(request_id)
            .and_then(|response| match &response.output {
                ResponseOutput::Embedding { normalized, .. } => Some(i32::from(*normalized)),
                ResponseOutput::Text(_) => None,
            })
            .unwrap_or(STATUS_FAILURE)
    }

    #[cfg(not(target_family = "wasm"))]
    fn completed_embedding_normalized(&self, _request_id: u32) -> i32 {
        STATUS_UNAVAILABLE
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
    fn media_marker(&self) -> Option<String> {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.media_marker().ok())
    }

    #[cfg(target_family = "wasm")]
    fn chat_template_source(&self) -> Option<String> {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.chat_template_source().ok().flatten())
    }

    #[cfg(target_family = "wasm")]
    fn bos_text(&self) -> Option<String> {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.get_bos_text().ok())
    }

    #[cfg(target_family = "wasm")]
    fn eos_text(&self) -> Option<String> {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.get_eos_text().ok())
    }

    #[cfg(target_family = "wasm")]
    fn probe_chat_boundary_info(&self) -> Option<String> {
        self.inner.runtime.as_ref().and_then(|runtime| {
            runtime
                .probe_chat_boundary_info()
                .ok()
                .and_then(|info| serde_json::to_string(&info).ok())
        })
    }

    #[cfg(target_family = "wasm")]
    fn token_buffer_ptr(&mut self) -> *mut u8 {
        self.inner.token_buffer.buffer.as_mut_ptr()
    }

    #[cfg(target_family = "wasm")]
    fn token_buffer_used_address(&mut self) -> *mut i32 {
        &mut self.inner.token_buffer.used
    }
}

#[cfg(target_family = "wasm")]
impl BrowserEngineInner {
    fn new() -> Self {
        Self {
            runtime: None,
            token_producer: None,
            token_consumer: None,
            token_buffer: TokenBuffer::new(TOKEN_BUFFER_CAPACITY),
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
struct TokenBuffer {
    buffer: Vec<u8>,
    used: i32,
}

#[cfg(target_family = "wasm")]
impl TokenBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0; capacity],
            used: 0,
        }
    }

    fn reset(&mut self) {
        self.used = 0;
    }

    fn write_frames(&mut self, frames: &[TokenRingFrame]) {
        let mut index = 0usize;
        while index < frames.len() {
            let stream_id = frames[index].stream_id;
            let sequence_start = frames[index].sequence;
            let mut byte_count = 0usize;
            let mut frame_count = 0u32;
            let mut end = index;
            while end < frames.len() && frames[end].stream_id == stream_id {
                byte_count = byte_count.saturating_add(frames[end].bytes.len());
                frame_count = frame_count.saturating_add(frames[end].frame_count);
                end += 1;
            }
            self.write_batch(
                stream_id,
                sequence_start,
                frame_count,
                &frames[index..end],
                byte_count,
            );
            index = end;
        }
    }

    fn write_batch(
        &mut self,
        request_id: u32,
        sequence_start: u32,
        frame_count: u32,
        frames: &[TokenRingFrame],
        byte_count: usize,
    ) {
        if request_id == 0 || frame_count == 0 || frames.is_empty() || byte_count == 0 {
            return;
        }
        let used = self.used.max(0) as usize;
        let record_len = TOKEN_RECORD_HEADER_BYTES + byte_count;
        let next_used = used + record_len;
        if next_used > self.buffer.len() {
            self.buffer.resize(next_used, 0);
        }
        let offset = used;
        self.buffer[offset..offset + 4].copy_from_slice(&request_id.to_le_bytes());
        self.buffer[offset + 4..offset + 8].copy_from_slice(&sequence_start.to_le_bytes());
        self.buffer[offset + 8..offset + 12].copy_from_slice(&frame_count.to_le_bytes());
        self.buffer[offset + 12..offset + 16].copy_from_slice(&(byte_count as u32).to_le_bytes());
        let mut payload_offset = offset + TOKEN_RECORD_HEADER_BYTES;
        for frame in frames {
            let next_payload_offset = payload_offset + frame.bytes.len();
            self.buffer[payload_offset..next_payload_offset].copy_from_slice(&frame.bytes);
            payload_offset = next_payload_offset;
        }
        self.used = (used + record_len) as i32;
    }
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_create() -> *mut BrowserEngine {
    catch_unwind(AssertUnwindSafe(|| {
        let id = NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed);
        Box::into_raw(Box::new(BrowserEngine::new(id)))
    }))
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_id(engine: *mut BrowserEngine) -> u32 {
    let Some(engine) = NonNull::new(engine) else {
        return 0;
    };
    unsafe { engine.as_ref().id }
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_load(
    engine: *mut BrowserEngine,
    model_path: *const c_char,
    runtime_config_json: *const c_char,
) -> i32 {
    with_engine_mut(engine, |engine| {
        engine.load(model_path, runtime_config_json)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_last_error_size(engine: *const BrowserEngine) -> i32 {
    with_engine_ref(engine, |engine| engine.last_error_size())
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_copy_last_error(
    engine: *const BrowserEngine,
    buffer: *mut u8,
    buffer_len: usize,
) -> i32 {
    with_engine_ref(engine, |engine| engine.copy_last_error(buffer, buffer_len))
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_close(engine: *mut BrowserEngine) -> i32 {
    catch_unwind(AssertUnwindSafe(|| {
        let Some(engine) = NonNull::new(engine) else {
            return STATUS_INVALID_ARGUMENTS;
        };
        unsafe {
            drop(Box::from_raw(engine.as_ptr()));
        }
        STATUS_OK
    }))
    .unwrap_or(STATUS_PANIC)
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_start_text_request(
    engine: *mut BrowserEngine,
    context_key: *const c_char,
    prompt: *const c_char,
    max_tokens: i32,
    emit_tokens: i32,
    grammar: *const c_char,
) -> u32 {
    with_engine_mut(engine, |engine| {
        engine.start_text_request(context_key, prompt, max_tokens, emit_tokens, grammar)
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn cogentlm_browser_engine_start_media_request(
    engine: *mut BrowserEngine,
    context_key: *const c_char,
    prompt: *const c_char,
    max_tokens: i32,
    image_count: i32,
    images_flat_buffer: *const u8,
    image_sizes: *const i32,
    emit_tokens: i32,
    grammar: *const c_char,
) -> u32 {
    with_engine_mut(engine, |engine| {
        engine.start_media_request(
            context_key,
            prompt,
            max_tokens,
            image_count,
            images_flat_buffer,
            image_sizes,
            emit_tokens,
            grammar,
        )
    })
}

#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn cogentlm_browser_engine_start_chat_request(
    engine: *mut BrowserEngine,
    context_key: *const c_char,
    messages_json: *const c_char,
    max_tokens: i32,
    image_count: i32,
    images_flat_buffer: *const u8,
    image_sizes: *const i32,
    emit_tokens: i32,
    grammar: *const c_char,
) -> u32 {
    with_engine_mut(engine, |engine| {
        engine.start_chat_request(
            context_key,
            messages_json,
            max_tokens,
            image_count,
            images_flat_buffer,
            image_sizes,
            emit_tokens,
            grammar,
        )
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_start_embedding_request(
    engine: *mut BrowserEngine,
    context_key: *const c_char,
    input: *const c_char,
    normalize: i32,
) -> u32 {
    with_engine_mut(engine, |engine| {
        engine.start_embedding_request(context_key, input, normalize)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_cancel_request(
    engine: *mut BrowserEngine,
    request_id: u32,
) -> i32 {
    with_engine_mut(engine, |engine| engine.cancel_request(request_id))
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_run_scheduler_loop(
    engine: *mut BrowserEngine,
    max_ticks: i32,
    max_completed_responses: i32,
    max_generated_tokens: i32,
    max_duration_us: i32,
    out: *mut BrowserSchedulerLoopResult,
) -> i32 {
    with_engine_mut(engine, |engine| {
        engine.run_scheduler_loop(
            max_ticks,
            max_completed_responses,
            max_generated_tokens,
            max_duration_us,
            out,
        )
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_completed_request_status(
    engine: *const BrowserEngine,
    request_id: u32,
) -> i32 {
    with_engine_ref(engine, |engine| engine.completed_status(request_id))
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_completed_request_output_kind(
    engine: *const BrowserEngine,
    request_id: u32,
) -> i32 {
    with_engine_ref(engine, |engine| engine.completed_output_kind(request_id))
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_completed_request_output_size(
    engine: *const BrowserEngine,
    request_id: u32,
) -> i32 {
    with_engine_ref(engine, |engine| {
        completed_output(engine, request_id)
            .map(|text| byte_len_i32(text.as_bytes()))
            .unwrap_or(STATUS_FAILURE)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_copy_completed_request_output(
    engine: *const BrowserEngine,
    request_id: u32,
    buffer: *mut u8,
    buffer_len: usize,
) -> i32 {
    with_engine_ref(engine, |engine| {
        completed_output(engine, request_id)
            .map(|text| copy_bytes_with_nul(text.as_bytes(), buffer, buffer_len))
            .unwrap_or(STATUS_FAILURE)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_completed_request_embedding_length(
    engine: *const BrowserEngine,
    request_id: u32,
) -> i32 {
    with_engine_ref(engine, |engine| engine.completed_embedding_len(request_id))
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_copy_completed_request_embedding(
    engine: *const BrowserEngine,
    request_id: u32,
    buffer: *mut f32,
    value_count: usize,
) -> i32 {
    with_engine_ref(engine, |engine| {
        engine.copy_completed_embedding(request_id, buffer, value_count)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_completed_request_embedding_pooling(
    engine: *const BrowserEngine,
    request_id: u32,
) -> i32 {
    with_engine_ref(engine, |engine| {
        engine.completed_embedding_pooling(request_id)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_completed_request_embedding_normalized(
    engine: *const BrowserEngine,
    request_id: u32,
) -> i32 {
    with_engine_ref(engine, |engine| {
        engine.completed_embedding_normalized(request_id)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_completed_request_error_size(
    engine: *const BrowserEngine,
    request_id: u32,
) -> i32 {
    with_engine_ref(engine, |engine| {
        completed_error(engine, request_id)
            .map(|text| byte_len_i32(text.as_bytes()))
            .unwrap_or(STATUS_FAILURE)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_copy_completed_request_error(
    engine: *const BrowserEngine,
    request_id: u32,
    buffer: *mut u8,
    buffer_len: usize,
) -> i32 {
    with_engine_ref(engine, |engine| {
        completed_error(engine, request_id)
            .map(|text| copy_bytes_with_nul(text.as_bytes(), buffer, buffer_len))
            .unwrap_or(STATUS_FAILURE)
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_consume_completed_request(
    engine: *mut BrowserEngine,
    request_id: u32,
) -> i32 {
    with_engine_mut(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            let Some(runtime) = engine.inner.runtime.as_mut() else {
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
            let _ = engine;
            let _ = request_id;
            STATUS_UNAVAILABLE
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_runtime_observability(
    engine: *const BrowserEngine,
    out: *mut BrowserRuntimeMetrics,
) -> i32 {
    if out.is_null() {
        return STATUS_INVALID_ARGUMENTS;
    }
    with_engine_ref(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            let Some(metrics) = engine.runtime_metrics() else {
                return STATUS_NOT_INITIALIZED;
            };
            unsafe {
                *out = metrics;
            }
            STATUS_OK
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            STATUS_UNAVAILABLE
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_completed_runtime_observability(
    engine: *const BrowserEngine,
    request_id: u32,
    out: *mut BrowserRuntimeMetrics,
) -> i32 {
    if out.is_null() {
        return STATUS_INVALID_ARGUMENTS;
    }
    with_engine_ref(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            let Some(metrics) = engine.completed_runtime_metrics(request_id) else {
                return STATUS_FAILURE;
            };
            unsafe {
                *out = metrics;
            }
            STATUS_OK
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            let _ = request_id;
            STATUS_UNAVAILABLE
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_token_buffer_pointer(
    engine: *mut BrowserEngine,
) -> *mut u8 {
    with_engine_mut(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            engine.token_buffer_ptr()
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            std::ptr::null_mut()
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_token_buffer_used_address(
    engine: *mut BrowserEngine,
) -> *mut i32 {
    with_engine_mut(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            engine.token_buffer_used_address()
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            std::ptr::null_mut()
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_media_marker(
    engine: *const BrowserEngine,
) -> *mut c_char {
    with_engine_ref(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            into_c_string(engine.media_marker().unwrap_or_default())
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            std::ptr::null_mut()
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_chat_template(
    engine: *const BrowserEngine,
) -> *mut c_char {
    with_engine_ref(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            into_c_string(engine.chat_template_source().unwrap_or_default())
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            std::ptr::null_mut()
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_bos_text(engine: *const BrowserEngine) -> *mut c_char {
    with_engine_ref(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            into_c_string(engine.bos_text().unwrap_or_default())
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            std::ptr::null_mut()
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_eos_text(engine: *const BrowserEngine) -> *mut c_char {
    with_engine_ref(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            into_c_string(engine.eos_text().unwrap_or_default())
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            std::ptr::null_mut()
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_probe_chat_boundary_info(
    engine: *const BrowserEngine,
) -> *mut c_char {
    with_engine_ref(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            into_c_string(engine.probe_chat_boundary_info().unwrap_or_default())
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            std::ptr::null_mut()
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_backend_observability_json(include_details: i32) -> *mut c_char {
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(target_family = "wasm")]
        {
            into_c_string(
                backend_observability_json(include_details != 0)
                    .unwrap_or_else(|_| "{}".to_string()),
            )
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = include_details;
            std::ptr::null_mut()
        }
    }))
    .unwrap_or(std::ptr::null_mut())
}

#[no_mangle]
/// # Safety
/// `value` must be null or a string pointer returned by this wasm module. Each
/// non-null pointer must be freed at most once.
pub unsafe extern "C" fn cogentlm_browser_engine_free_string(value: *mut c_char) {
    unsafe { free_c_string(value) }
}

fn with_engine_mut<T>(engine: *mut BrowserEngine, f: impl FnOnce(&mut BrowserEngine) -> T) -> T
where
    T: Default,
{
    catch_unwind(AssertUnwindSafe(|| {
        let Some(mut engine) = NonNull::new(engine) else {
            return T::default();
        };
        f(unsafe { engine.as_mut() })
    }))
    .unwrap_or_default()
}

fn with_engine_ref<T>(engine: *const BrowserEngine, f: impl FnOnce(&BrowserEngine) -> T) -> T
where
    T: Default,
{
    catch_unwind(AssertUnwindSafe(|| {
        let Some(engine) = NonNull::new(engine.cast_mut()) else {
            return T::default();
        };
        f(unsafe { engine.as_ref() })
    }))
    .unwrap_or_default()
}

#[cfg(target_family = "wasm")]
fn read_c_string(value: *const c_char) -> Option<String> {
    read_optional_c_string(value).filter(|value| !value.is_empty())
}

fn byte_len_i32(bytes: &[u8]) -> i32 {
    i32::try_from(bytes.len()).unwrap_or(STATUS_FAILURE)
}

#[cfg(target_family = "wasm")]
fn value_len_i32(len: usize) -> Option<i32> {
    i32::try_from(len).ok()
}

fn copy_bytes_with_nul(bytes: &[u8], buffer: *mut u8, buffer_len: usize) -> i32 {
    if buffer.is_null() || buffer_len == 0 || buffer_len <= bytes.len() {
        return STATUS_INVALID_ARGUMENTS;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), buffer, bytes.len());
        *buffer.add(bytes.len()) = 0;
    }
    byte_len_i32(bytes)
}

#[cfg(target_family = "wasm")]
fn read_runtime_config(runtime_config_json: *const c_char) -> Result<NativeRuntimeConfig, String> {
    let Some(raw) = read_optional_c_string(runtime_config_json) else {
        return Err("runtime config JSON is not valid UTF-8".to_string());
    };
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
fn copy_image_buffers(
    image_count: i32,
    images_flat_buffer: *const u8,
    image_sizes: *const i32,
) -> Option<Vec<Vec<u8>>> {
    if image_count <= 0 || images_flat_buffer.is_null() || image_sizes.is_null() {
        return None;
    }
    let image_count = usize::try_from(image_count).ok()?;
    let sizes = unsafe { std::slice::from_raw_parts(image_sizes, image_count) };
    let total_bytes = sizes.iter().try_fold(0usize, |sum, size| {
        let size = usize::try_from(*size).ok()?;
        sum.checked_add(size)
    })?;
    let flat = unsafe { std::slice::from_raw_parts(images_flat_buffer, total_bytes) };
    let mut images = Vec::with_capacity(image_count);
    let mut offset = 0usize;
    for size in sizes {
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
