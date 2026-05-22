#[cfg(target_family = "wasm")]
use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};

#[cfg(target_family = "wasm")]
use std::time::{Duration, Instant};

#[cfg(target_family = "wasm")]
use cogentlm_engine::backend::{backend_observability_json, set_llama_log_quiet};
#[cfg(target_family = "wasm")]
use cogentlm_engine::runtime::config::NativeRuntimeConfig;
#[cfg(target_family = "wasm")]
use cogentlm_engine::runtime::request::{
    token_byte_ring, GenerateResponse, GenerateResponseStatus, GenerateTokenEmissionMode,
    TokenByteRingConsumer, TokenByteRingProducer, TOKEN_RING_DEFAULT_CAPACITY,
};
#[cfg(target_family = "wasm")]
use cogentlm_engine::runtime::{InferenceRuntime, RequestStepResult, SchedulerBurstResult};

// Provided by `wasm_exports.cpp`. Calls back into the host JS to drain the
// streaming buffer into the SAB ring. Synchronous; the worker thread blocks
// inside it for a few microseconds.
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

const ABI_VERSION: u32 = 3;

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
const STREAMING_BUFFER_CAPACITY: usize = 256 * 1024;
#[cfg(target_family = "wasm")]
const STREAMING_RECORD_HEADER_BYTES: usize = 8;

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
    token_ring_drop_count: u64,
    streaming_buffer: StreamingBuffer,
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
        self.inner.streaming_buffer.reset();
        self.inner.token_ring_drop_count = 0;
    }

    #[cfg(target_family = "wasm")]
    fn start_text_request(
        &mut self,
        context_key: *const c_char,
        prompt: *const c_char,
        max_tokens: i32,
        token_emission_mode: i32,
        grammar: *const c_char,
    ) -> u32 {
        let Some(prompt) = read_c_string(prompt) else {
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
            token_emission_mode,
        )
    }

    #[cfg(not(target_family = "wasm"))]
    fn start_text_request(
        &mut self,
        _context_key: *const c_char,
        _prompt: *const c_char,
        _max_tokens: i32,
        _token_emission_mode: i32,
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
        token_emission_mode: i32,
        grammar: *const c_char,
    ) -> u32 {
        let Some(prompt) = read_c_string(prompt) else {
            return 0;
        };
        let Some(images) = copy_image_buffers(image_count, images_flat_buffer, image_sizes) else {
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
            token_emission_mode,
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
        _token_emission_mode: i32,
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
        token_emission_mode: i32,
        grammar: *const c_char,
    ) -> u32 {
        let Some(runtime) = self.inner.runtime.as_ref() else {
            return 0;
        };
        let Some(messages_json) = read_c_string(messages_json) else {
            return 0;
        };
        let Ok(prompt) = runtime.apply_chat_template_json(&messages_json, true) else {
            return 0;
        };
        if prompt.is_empty() {
            return 0;
        }
        let images = if image_count > 0 {
            let Some(images) = copy_image_buffers(image_count, images_flat_buffer, image_sizes)
            else {
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
            token_emission_mode,
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
        _token_emission_mode: i32,
        _grammar: *const c_char,
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
        let request_id = if images.is_empty() {
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
        }
        .unwrap_or_default();
        if request_id != 0 {
            if let Some(producer) = self.inner.token_producer.as_ref() {
                runtime.add_token_ring_producer(request_id, producer.clone());
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
        max_emitted_tokens: i32,
        max_duration_us: i32,
        streaming_active: bool,
        out: *mut BrowserSchedulerLoopResult,
        run_continuous_loop: bool,
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

        // For one-shot bursts (JS calls cogentlm_browser_engine_run_scheduler_burst)
        // keep the original behavior: single call, drain, return.
        if !run_continuous_loop {
            let runtime = self
                .inner
                .runtime
                .as_mut()
                .expect("runtime present after early return");
            let burst = runtime.run_scheduler_burst(
                max_ticks,
                max_completed_responses,
                max_emitted_tokens,
                duration,
            );
            self.drain_token_ring();
            unsafe {
                *out = scheduler_loop_result_from_runtime(burst);
            }
            return burst.status as i32;
        }

        // Bulk (non-streaming) path: just run the runtime's own monolithic
        // loop. No per-tick drain/yield overhead — matches the pre-port
        // C++ baseline. The host won't peek at the streaming buffer mid-loop
        // for non-streaming requests anyway, so there's nothing to drain in
        // between.
        if !streaming_active {
            let runtime = self
                .inner
                .runtime
                .as_mut()
                .expect("runtime present after early return");
            let burst = runtime.run_scheduler_loop(
                max_ticks,
                max_completed_responses,
                max_emitted_tokens,
                duration,
            );
            self.drain_token_ring();
            unsafe {
                *out = scheduler_loop_result_from_runtime(burst);
            }
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

            let runtime = self
                .inner
                .runtime
                .as_mut()
                .expect("runtime present for the duration of the burst loop");
            let burst = runtime.run_scheduler_burst(
                step_ticks,
                remaining_completed,
                step_emitted_budget,
                step_duration,
            );
            acc.ticks_executed += burst.ticks_executed;
            acc.progressed_ticks += burst.progressed_ticks;
            acc.completed_response_count += burst.completed_response_count;
            acc.emitted_token_count += burst.emitted_token_count;

            // Drain only when there's something to drain. Prefill bursts that
            // didn't emit a token leave the ring empty; skipping the drain
            // saves a mutex acquisition + Vec setup per prefill burst.
            if burst.emitted_token_count > 0 {
                self.drain_token_ring();
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
        unsafe {
            *out = scheduler_loop_result_from_runtime(acc);
        }
        acc.status as i32
    }

    #[cfg(not(target_family = "wasm"))]
    fn run_scheduler_loop(
        &mut self,
        _max_ticks: i32,
        _max_completed_responses: i32,
        _max_emitted_tokens: i32,
        _max_duration_us: i32,
        _streaming_active: bool,
        _out: *mut BrowserSchedulerLoopResult,
        _run_continuous_loop: bool,
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
    fn completed_status(&self, request_id: u32) -> i32 {
        let Some(runtime) = self.inner.runtime.as_ref() else {
            return COMPLETED_REQUEST_STATUS_UNKNOWN;
        };
        if let Some(response) = runtime.try_peek_completed_response(request_id) {
            return match response.status {
                GenerateResponseStatus::Pending => COMPLETED_REQUEST_STATUS_PENDING,
                GenerateResponseStatus::Completed => COMPLETED_REQUEST_STATUS_COMPLETED,
                GenerateResponseStatus::Cancelled => COMPLETED_REQUEST_STATUS_CANCELLED,
                GenerateResponseStatus::Failed => COMPLETED_REQUEST_STATUS_FAILED,
            };
        }
        if runtime.has_request(request_id) {
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
    fn completed_response(&self, request_id: u32) -> Option<GenerateResponse> {
        self.inner
            .runtime
            .as_ref()
            .and_then(|runtime| runtime.try_peek_completed_response(request_id))
    }

    #[cfg(target_family = "wasm")]
    fn consume_completed_response(&mut self, request_id: u32) -> i32 {
        let Some(runtime) = self.inner.runtime.as_mut() else {
            return 0;
        };
        runtime.remove_token_ring_producer(request_id);
        i32::from(runtime.consume_completed_response(request_id))
    }

    #[cfg(not(target_family = "wasm"))]
    fn consume_completed_response(&mut self, _request_id: u32) -> i32 {
        0
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
        self.completed_response(request_id)
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
    fn streaming_buffer_ptr(&mut self) -> *mut u8 {
        self.inner.streaming_buffer.buffer.as_mut_ptr()
    }

    #[cfg(target_family = "wasm")]
    fn streaming_buffer_capacity(&self) -> i32 {
        self.inner.streaming_buffer.buffer.len() as i32
    }

    #[cfg(target_family = "wasm")]
    fn streaming_buffer_used_address(&mut self) -> *mut i32 {
        &mut self.inner.streaming_buffer.used
    }

    #[cfg(target_family = "wasm")]
    fn streaming_buffer_drop_count_address(&mut self) -> *mut i32 {
        &mut self.inner.streaming_buffer.drop_count
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
    token_emission_mode: i32,
    grammar: *const c_char,
) -> u32 {
    with_engine_mut(engine, |engine| {
        engine.start_text_request(
            context_key,
            prompt,
            max_tokens,
            token_emission_mode,
            grammar,
        )
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
    token_emission_mode: i32,
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
            token_emission_mode,
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
    token_emission_mode: i32,
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
            token_emission_mode,
            grammar,
        )
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
    max_emitted_tokens: i32,
    max_duration_us: i32,
    streaming_active: i32,
    out: *mut BrowserSchedulerLoopResult,
) -> i32 {
    with_engine_mut(engine, |engine| {
        engine.run_scheduler_loop(
            max_ticks,
            max_completed_responses,
            max_emitted_tokens,
            max_duration_us,
            streaming_active != 0,
            out,
            true,
        )
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_run_scheduler_burst(
    engine: *mut BrowserEngine,
    max_ticks: i32,
    max_completed_responses: i32,
    max_emitted_tokens: i32,
    max_duration_us: i32,
    out: *mut BrowserSchedulerLoopResult,
) -> i32 {
    with_engine_mut(engine, |engine| {
        engine.run_scheduler_loop(
            max_ticks,
            max_completed_responses,
            max_emitted_tokens,
            max_duration_us,
            false,
            out,
            false,
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
        engine.consume_completed_response(request_id)
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
pub extern "C" fn cogentlm_browser_engine_streaming_buffer_pointer(
    engine: *mut BrowserEngine,
) -> *mut u8 {
    with_engine_mut(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            engine.streaming_buffer_ptr()
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            std::ptr::null_mut()
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_streaming_buffer_capacity(
    engine: *const BrowserEngine,
) -> i32 {
    with_engine_ref(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            engine.streaming_buffer_capacity()
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            0
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_streaming_buffer_used_address(
    engine: *mut BrowserEngine,
) -> *mut i32 {
    with_engine_mut(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            engine.streaming_buffer_used_address()
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = engine;
            std::ptr::null_mut()
        }
    })
}

#[no_mangle]
pub extern "C" fn cogentlm_browser_engine_streaming_buffer_drop_count_address(
    engine: *mut BrowserEngine,
) -> *mut i32 {
    with_engine_mut(engine, |engine| {
        #[cfg(target_family = "wasm")]
        {
            engine.streaming_buffer_drop_count_address()
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
pub extern "C" fn cogentlm_browser_engine_free_string(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(value));
    }
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
    if value.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned(),
    )
    .filter(|value| !value.is_empty())
}

#[cfg(target_family = "wasm")]
fn read_optional_c_string(value: *const c_char) -> Option<String> {
    if value.is_null() {
        return Some(String::new());
    }
    Some(
        unsafe { CStr::from_ptr(value) }
            .to_string_lossy()
            .into_owned(),
    )
}

#[cfg(target_family = "wasm")]
fn into_c_string(value: String) -> *mut c_char {
    let sanitized = value.replace('\0', "");
    CString::new(sanitized)
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

fn byte_len_i32(bytes: &[u8]) -> i32 {
    i32::try_from(bytes.len()).unwrap_or(STATUS_FAILURE)
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
fn emission_mode(token_emission_mode: i32) -> GenerateTokenEmissionMode {
    if token_emission_mode == 1 {
        GenerateTokenEmissionMode::TokenStream
    } else {
        GenerateTokenEmissionMode::None
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
fn completed_output(engine: &BrowserEngine, request_id: u32) -> Option<String> {
    engine
        .completed_response(request_id)
        .map(|response| response.output_text)
}

#[cfg(not(target_family = "wasm"))]
fn completed_output(_engine: &BrowserEngine, _request_id: u32) -> Option<String> {
    None
}

#[cfg(target_family = "wasm")]
fn completed_error(engine: &BrowserEngine, request_id: u32) -> Option<String> {
    engine
        .completed_response(request_id)
        .map(|response| response.error_message)
}

#[cfg(not(target_family = "wasm"))]
fn completed_error(_engine: &BrowserEngine, _request_id: u32) -> Option<String> {
    None
}

#[cfg(test)]
#[path = "tests/root_tests.rs"]
mod root_tests;
