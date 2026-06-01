use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::{env, fs, ptr, slice};

use cogentlm_engine::backend::backend_observability_json;
use serde_json::{json, Value};

use crate::engine::{BrowserEngine, ABI_VERSION};
use crate::hash::BrowserSha256Hasher;
use crate::ingest::{
    GgufCloseShardCallback, GgufOpenShardCallback, GgufReadAtCallback, GgufWriteShardCallback,
};
use crate::{BrowserRuntimeMetrics, BrowserSchedulerLoopResult};

const STATUS_OK: i32 = 0;
const STATUS_FAILURE: i32 = -1;
const STATUS_INVALID_ARGUMENTS: i32 = -2;
const COMPLETED_REQUEST_STATUS_UNKNOWN: i32 = 4;
const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_991.0;
const LLAMA_CACHE_DIR: &str = "/tmp/cogentlm-llama-cache";

thread_local! {
    static CURRENT_ENGINE: RefCell<Option<Box<BrowserEngine>>> = RefCell::new(None);
    static LAST_ENGINE_ERROR: RefCell<String> = RefCell::new(String::new());
    static MEDIA_MARKER_CACHE: RefCell<Option<CString>> = RefCell::new(None);
    static CHAT_TEMPLATE_CACHE: RefCell<Option<CString>> = RefCell::new(None);
}

#[no_mangle]
pub extern "C" fn CE_RustBrowserEngineAbiVersion() -> i32 {
    ABI_VERSION as i32
}

#[no_mangle]
pub extern "C" fn CE_RustBrowserEngineCreate() -> usize {
    Box::into_raw(Box::new(BrowserEngine::create())) as usize
}

#[no_mangle]
pub unsafe extern "C" fn CE_RustBrowserEngineId(engine: usize) -> i32 {
    if engine == 0 {
        return 0;
    }
    (*(engine as *const BrowserEngine)).id() as i32
}

#[no_mangle]
pub unsafe extern "C" fn CE_RustBrowserEngineClose(engine: usize) -> i32 {
    if engine == 0 {
        return STATUS_INVALID_ARGUMENTS;
    }
    drop(Box::from_raw(engine as *mut BrowserEngine));
    STATUS_OK
}

#[no_mangle]
pub extern "C" fn CE_BrowserCacheLayout(
    source_bytes: f64,
    source_bytes_known: i32,
    direct_load_max_bytes: f64,
    shard_max_bytes: f64,
) -> i32 {
    let Some(source_bytes) = read_size_arg(source_bytes) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    let Some(direct_load_max_bytes) = read_size_arg(direct_load_max_bytes) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    let Some(shard_max_bytes) = read_size_arg(shard_max_bytes) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    crate::ingest::browser_cache_layout(
        source_bytes,
        source_bytes_known != 0,
        direct_load_max_bytes,
        shard_max_bytes,
    )
}

#[no_mangle]
pub unsafe extern "C" fn CE_GgufPlanSplitCount(
    source_bytes: f64,
    shard_max_bytes: f64,
    user_data: *mut c_void,
    read_at: Option<GgufReadAtCallback>,
) -> i32 {
    let Some(source_bytes) = read_size_arg(source_bytes) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    let Some(shard_max_bytes) = read_size_arg(shard_max_bytes) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    let Some(read_at) = read_at else {
        return STATUS_INVALID_ARGUMENTS;
    };
    crate::ingest::gguf_plan_split_count(source_bytes, shard_max_bytes, user_data, read_at)
}

#[no_mangle]
pub unsafe extern "C" fn CE_GgufSplitStream(
    source_bytes: f64,
    output_prefix: *const c_char,
    shard_max_bytes: f64,
    user_data: *mut c_void,
    read_at: Option<GgufReadAtCallback>,
    open_shard: Option<GgufOpenShardCallback>,
    write_shard: Option<GgufWriteShardCallback>,
    close_shard: Option<GgufCloseShardCallback>,
) -> i32 {
    let Some(source_bytes) = read_size_arg(source_bytes) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    let Some(shard_max_bytes) = read_size_arg(shard_max_bytes) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    let Some(output_prefix) = required_cstr(output_prefix) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    let (Some(read_at), Some(open_shard), Some(write_shard), Some(close_shard)) =
        (read_at, open_shard, write_shard, close_shard)
    else {
        return STATUS_INVALID_ARGUMENTS;
    };
    crate::ingest::gguf_split_stream(
        source_bytes,
        &output_prefix,
        shard_max_bytes,
        user_data,
        read_at,
        open_shard,
        write_shard,
        close_shard,
    )
}

#[no_mangle]
pub unsafe extern "C" fn CE_DetectModelFromGgufBytes(
    name: *const c_char,
    bytes: *const u8,
    bytes_len: f64,
) -> *mut c_char {
    let Some(name) = required_cstr(name) else {
        return owned_json_error("INVALID_GGUF", "invalid GGUF byte length");
    };
    let Some(bytes_len) = read_pointer_size_arg(bytes_len) else {
        return owned_json_error("INVALID_GGUF", "invalid GGUF byte length");
    };
    let Some(bytes) = bytes_from_raw(bytes, bytes_len) else {
        return owned_json_error("INVALID_GGUF", "invalid GGUF byte length");
    };
    owned_string(crate::gguf::detect_model_from_gguf_bytes_json(&name, bytes))
}

#[no_mangle]
pub extern "C" fn CE_Sha256Create() -> usize {
    Box::into_raw(Box::new(BrowserSha256Hasher::new())) as usize
}

#[no_mangle]
pub unsafe extern "C" fn CE_Sha256Update(hasher: usize, bytes: *const u8, bytes_len: f64) -> i32 {
    if hasher == 0 {
        return STATUS_INVALID_ARGUMENTS;
    }
    let Some(bytes_len) = read_pointer_size_arg(bytes_len) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    let Some(bytes) = bytes_from_raw(bytes, bytes_len) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    (*(hasher as *mut BrowserSha256Hasher)).update(bytes);
    STATUS_OK
}

#[no_mangle]
pub unsafe extern "C" fn CE_Sha256Finalize(hasher: usize) -> *mut c_char {
    if hasher == 0 {
        return ptr::null_mut();
    }
    let hasher = Box::from_raw(hasher as *mut BrowserSha256Hasher);
    owned_string(hasher.finalize_hex())
}

#[no_mangle]
pub unsafe extern "C" fn CE_Sha256Close(hasher: usize) -> i32 {
    if hasher == 0 {
        return STATUS_INVALID_ARGUMENTS;
    }
    drop(Box::from_raw(hasher as *mut BrowserSha256Hasher));
    1
}

#[no_mangle]
pub unsafe extern "C" fn CE_ModelServiceCreate(config_json: *const c_char) -> *mut c_char {
    let Some(config_json) = required_cstr(config_json) else {
        return owned_json_error("INVALID_MODEL_SOURCE", "service config JSON is missing");
    };
    owned_string(crate::lifecycle::model_service_create_json(&config_json))
}

#[no_mangle]
pub extern "C" fn CE_ModelServiceClose(service: usize) -> i32 {
    crate::lifecycle::model_service_close(service)
}

#[no_mangle]
pub extern "C" fn CE_ModelServiceList(service: usize) -> *mut c_char {
    owned_string(crate::lifecycle::model_service_list_json(service))
}

#[no_mangle]
pub extern "C" fn CE_ModelServiceCurrent(service: usize) -> *mut c_char {
    owned_string(crate::lifecycle::model_service_current_json(service))
}

#[no_mangle]
pub extern "C" fn CE_ModelServiceManifest(service: usize) -> *mut c_char {
    owned_string(crate::lifecycle::model_service_manifest_json(service))
}

#[no_mangle]
pub unsafe extern "C" fn CE_ModelServicePrepareLoad(
    service: usize,
    source_json: *const c_char,
    options_json: *const c_char,
) -> *mut c_char {
    let (Some(source_json), Some(options_json)) =
        (required_cstr(source_json), required_cstr(options_json))
    else {
        return owned_json_error(
            "INVALID_MODEL_SOURCE",
            "load source or options JSON is missing",
        );
    };
    owned_string(crate::lifecycle::model_service_prepare_load_json(
        service,
        &source_json,
        &options_json,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn CE_ModelServiceCommitLoad(
    service: usize,
    commit_json: *const c_char,
) -> *mut c_char {
    let Some(commit_json) = required_cstr(commit_json) else {
        return owned_json_error("INVALID_MODEL_SOURCE", "load commit JSON is missing");
    };
    owned_string(crate::lifecycle::model_service_commit_load_json(
        service,
        &commit_json,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn CE_ModelServiceAbortLoad(
    service: usize,
    error_json: *const c_char,
) -> *mut c_char {
    owned_string(crate::lifecycle::model_service_abort_load_json(
        service,
        &optional_cstr(error_json),
    ))
}

#[no_mangle]
pub unsafe extern "C" fn CE_ModelServiceRemove(
    service: usize,
    model_id: *const c_char,
) -> *mut c_char {
    let Some(model_id) = required_cstr(model_id) else {
        return owned_json_error("INVALID_MODEL_SOURCE", "model id is missing");
    };
    owned_string(crate::lifecycle::model_service_remove_json(
        service, &model_id,
    ))
}

#[no_mangle]
pub extern "C" fn CE_ModelServiceUnload(service: usize) -> *mut c_char {
    owned_string(crate::lifecycle::model_service_unload_json(service))
}

#[no_mangle]
pub extern "C" fn CE_ModelServiceSnapshot(service: usize) -> *mut c_char {
    owned_string(crate::lifecycle::model_service_snapshot_json(service))
}

#[no_mangle]
pub extern "C" fn CE_ModelServiceDrainEvents(service: usize) -> *mut c_char {
    owned_string(crate::lifecycle::model_service_drain_events_json(service))
}

#[no_mangle]
pub unsafe extern "C" fn CE_ModelServiceRecordEvent(
    service: usize,
    event_type: *const c_char,
    patch_json: *const c_char,
) -> *mut c_char {
    let (Some(event_type), Some(patch_json)) =
        (required_cstr(event_type), required_cstr(patch_json))
    else {
        return owned_json_error(
            "INVALID_MODEL_SOURCE",
            "event type or patch JSON is missing",
        );
    };
    owned_string(crate::lifecycle::model_service_record_event_json(
        service,
        &event_type,
        &patch_json,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn CE_Init(
    model_path: *const c_char,
    runtime_config_json: *const c_char,
) -> i32 {
    let (Some(model_path), Some(runtime_config_json)) = (
        required_cstr(model_path),
        required_cstr(runtime_config_json),
    ) else {
        set_last_engine_error("engine init received a null string");
        return STATUS_INVALID_ARGUMENTS;
    };

    close_current_engine();
    ensure_llama_cache_env();

    let mut engine = Box::new(BrowserEngine::create());
    let status = engine.load(&model_path, &runtime_config_json);
    if status != STATUS_OK {
        let message = if engine.last_error().is_empty() {
            "Rust browser engine returned failure during load".to_string()
        } else {
            engine.last_error().to_string()
        };
        set_last_engine_error(message);
        return status;
    }

    CURRENT_ENGINE.with(|current| {
        *current.borrow_mut() = Some(engine);
    });
    clear_last_engine_error();
    STATUS_OK
}

#[no_mangle]
pub extern "C" fn CE_GetLastEngineErrorSize() -> i32 {
    LAST_ENGINE_ERROR.with(|error| byte_len_i32(error.borrow().as_bytes()))
}

#[no_mangle]
pub unsafe extern "C" fn CE_CopyLastEngineError(buffer: *mut c_char, capacity: i32) -> i32 {
    let Some(buffer_len) = read_nonnegative_count(capacity) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    if buffer.is_null() {
        return STATUS_INVALID_ARGUMENTS;
    }
    let buffer = slice::from_raw_parts_mut(buffer as *mut u8, buffer_len);
    LAST_ENGINE_ERROR.with(|error| copy_bytes_with_nul(error.borrow().as_bytes(), buffer))
}

#[no_mangle]
pub extern "C" fn CE_Close() {
    close_current_engine();
}

#[no_mangle]
pub extern "C" fn CE_GetBackendObservabilityJson() -> *mut c_char {
    owned_string(enriched_backend_observability_json())
}

#[no_mangle]
pub extern "C" fn CE_GetMediaMarker() -> *const c_char {
    with_current_engine(ptr::null(), |engine| {
        cache_c_string(&MEDIA_MARKER_CACHE, engine.media_marker())
    })
}

#[no_mangle]
pub extern "C" fn CE_GetChatTemplate() -> *const c_char {
    with_current_engine(ptr::null(), |engine| {
        cache_c_string(&CHAT_TEMPLATE_CACHE, engine.chat_template_source())
    })
}

#[no_mangle]
pub extern "C" fn CE_GetBosText() -> *mut c_char {
    owned_string(with_current_engine(String::new(), BrowserEngine::bos_text))
}

#[no_mangle]
pub extern "C" fn CE_GetEosText() -> *mut c_char {
    owned_string(with_current_engine(String::new(), BrowserEngine::eos_text))
}

#[no_mangle]
pub unsafe extern "C" fn CE_PairingValidate(
    classified_json: *const c_char,
    explicit_projector_id: *const c_char,
) -> *mut c_char {
    let Some(classified_json) = required_cstr(classified_json) else {
        return owned_json_error("INVALID_MODEL_SOURCE", "classified asset JSON is missing");
    };
    owned_string(crate::pairing::pairing_validate_json(
        &classified_json,
        &optional_cstr(explicit_projector_id),
    ))
}

#[no_mangle]
pub extern "C" fn CE_ProbeChatBoundaryInfo() -> *mut c_char {
    owned_string(with_current_engine(
        String::new(),
        BrowserEngine::probe_chat_boundary_info,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartTextRequestWithTokenEmissionMode(
    context_key: *const c_char,
    prompt: *const c_char,
    n_tokens: i32,
    token_emission_mode: i32,
    grammar: *const c_char,
) -> u32 {
    if prompt.is_null() || !is_valid_prediction_tokens(n_tokens) {
        return 0;
    }
    let context_key = optional_cstr(context_key);
    let prompt = optional_cstr(prompt);
    let grammar = optional_cstr(grammar);
    with_current_engine_mut(0, |engine| {
        let request_id = engine.start_text_request(
            &context_key,
            &prompt,
            n_tokens,
            token_emission_mode,
            &grammar,
        );
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartMediaRequestWithTokenEmissionMode(
    context_key: *const c_char,
    prompt: *const c_char,
    n_tokens: i32,
    n_images: i32,
    images_flat_buffer: *const u8,
    image_sizes: *const i32,
    token_emission_mode: i32,
    grammar: *const c_char,
) -> u32 {
    if prompt.is_null() || !is_valid_prediction_tokens(n_tokens) {
        return 0;
    }
    let Some((images, sizes)) = media_slices(images_flat_buffer, image_sizes, n_images) else {
        set_last_engine_error("media buffers are invalid");
        return 0;
    };
    let context_key = optional_cstr(context_key);
    let prompt = optional_cstr(prompt);
    let grammar = optional_cstr(grammar);
    with_current_engine_mut(0, |engine| {
        let request_id = engine.start_media_request(
            &context_key,
            &prompt,
            n_tokens,
            images,
            sizes,
            token_emission_mode,
            &grammar,
        );
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartChatRequestWithTokenEmissionMode(
    context_key: *const c_char,
    messages_json: *const c_char,
    n_tokens: i32,
    n_images: i32,
    images_flat_buffer: *const u8,
    image_sizes: *const i32,
    token_emission_mode: i32,
    grammar: *const c_char,
) -> u32 {
    if messages_json.is_null() || !is_valid_prediction_tokens(n_tokens) {
        return 0;
    }
    let Some((images, sizes)) = media_slices(images_flat_buffer, image_sizes, n_images) else {
        set_last_engine_error("media buffers are invalid");
        return 0;
    };
    let context_key = optional_cstr(context_key);
    let messages_json = optional_cstr(messages_json);
    let grammar = optional_cstr(grammar);
    with_current_engine_mut(0, |engine| {
        let request_id = engine.start_chat_request(
            &context_key,
            &messages_json,
            n_tokens,
            images,
            sizes,
            token_emission_mode,
            &grammar,
        );
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartEmbeddingRequest(
    context_key: *const c_char,
    input: *const c_char,
    normalize: i32,
) -> u32 {
    if input.is_null() {
        return 0;
    }
    let context_key = optional_cstr(context_key);
    let input = optional_cstr(input);
    with_current_engine_mut(0, |engine| {
        let request_id = engine.start_embedding_request(&context_key, &input, normalize);
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub extern "C" fn CE_CancelRequest(request_id: u32) -> i32 {
    if request_id == 0 {
        return 0;
    }
    with_current_engine_mut(0, |engine| engine.cancel_request(request_id))
}

#[no_mangle]
pub unsafe extern "C" fn CE_GetRuntimeObservability(
    out_metrics: *mut BrowserRuntimeMetrics,
) -> i32 {
    if out_metrics.is_null() {
        return STATUS_FAILURE;
    }
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.runtime_observability(&mut *out_metrics)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_RunSchedulerLoop(
    max_ticks: i32,
    max_completed_responses: i32,
    max_emitted_tokens: i32,
    max_duration_us: i32,
    streaming_active: i32,
    out_result: *mut BrowserSchedulerLoopResult,
) -> i32 {
    if out_result.is_null() {
        return STATUS_FAILURE;
    }
    with_current_engine_mut(STATUS_FAILURE, |engine| {
        engine.run_scheduler_loop(
            max_ticks,
            max_completed_responses,
            max_emitted_tokens,
            max_duration_us,
            streaming_active != 0,
            &mut *out_result,
        )
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestStatus(request_id: u32) -> i32 {
    if request_id == 0 {
        return COMPLETED_REQUEST_STATUS_UNKNOWN;
    }
    with_current_engine(COMPLETED_REQUEST_STATUS_UNKNOWN, |engine| {
        engine.completed_status(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestOutputKind(request_id: u32) -> i32 {
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.completed_output_kind(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetStreamingBufferPointer() -> *const u8 {
    with_current_engine_mut(ptr::null(), |engine| {
        engine.streaming_buffer_ptr() as *const u8
    })
}

#[no_mangle]
pub extern "C" fn CE_GetStreamingBufferUsedAddress() -> *mut i32 {
    with_current_engine_mut(ptr::null_mut(), |engine| {
        engine.streaming_buffer_used_address() as *mut i32
    })
}

#[no_mangle]
pub extern "C" fn CE_GetStreamingBufferDropCountAddress() -> *mut i32 {
    with_current_engine_mut(ptr::null_mut(), |engine| {
        engine.streaming_buffer_drop_count_address() as *mut i32
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestOutputSize(request_id: u32) -> i32 {
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.completed_output_size(request_id)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_CopyCompletedRequestOutput(
    request_id: u32,
    buffer: *mut c_char,
    capacity: i32,
) -> i32 {
    let Some(buffer) = mutable_u8_slice(buffer as *mut u8, capacity) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.copy_completed_output(request_id, buffer)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestEmbeddingLength(request_id: u32) -> i32 {
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.completed_embedding_len(request_id)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_CopyCompletedRequestEmbedding(
    request_id: u32,
    buffer: *mut f32,
    value_count: i32,
) -> i32 {
    let Some(value_count) = read_nonnegative_count(value_count) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    if buffer.is_null() {
        return STATUS_INVALID_ARGUMENTS;
    }
    let buffer = slice::from_raw_parts_mut(buffer, value_count);
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.copy_completed_embedding(request_id, buffer)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestEmbeddingPooling(request_id: u32) -> i32 {
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.completed_embedding_pooling(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestEmbeddingNormalized(request_id: u32) -> i32 {
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.completed_embedding_normalized(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestErrorSize(request_id: u32) -> i32 {
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.completed_error_size(request_id)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_CopyCompletedRequestError(
    request_id: u32,
    buffer: *mut c_char,
    capacity: i32,
) -> i32 {
    let Some(buffer) = mutable_u8_slice(buffer as *mut u8, capacity) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.copy_completed_error(request_id, buffer)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_GetCompletedRequestRuntimeObservability(
    request_id: u32,
    out_metrics: *mut BrowserRuntimeMetrics,
) -> i32 {
    if out_metrics.is_null() {
        return STATUS_FAILURE;
    }
    with_current_engine(STATUS_FAILURE, |engine| {
        engine.completed_runtime_observability(request_id, &mut *out_metrics)
    })
}

#[no_mangle]
pub extern "C" fn CE_ConsumeCompletedRequest(request_id: u32) -> i32 {
    with_current_engine_mut(0, |engine| engine.consume_completed_request(request_id))
}

#[no_mangle]
pub unsafe extern "C" fn CE_FreeString(value: *mut c_char) {
    if !value.is_null() {
        drop(CString::from_raw(value));
    }
}

fn with_current_engine<T>(fallback: T, operation: impl FnOnce(&BrowserEngine) -> T) -> T {
    CURRENT_ENGINE.with(|current| {
        let current = current.borrow();
        current.as_deref().map(operation).unwrap_or(fallback)
    })
}

fn with_current_engine_mut<T>(fallback: T, operation: impl FnOnce(&mut BrowserEngine) -> T) -> T {
    CURRENT_ENGINE.with(|current| {
        let mut current = current.borrow_mut();
        current.as_deref_mut().map(operation).unwrap_or(fallback)
    })
}

fn close_current_engine() {
    CURRENT_ENGINE.with(|current| {
        *current.borrow_mut() = None;
    });
    MEDIA_MARKER_CACHE.with(|cache| {
        *cache.borrow_mut() = None;
    });
    CHAT_TEMPLATE_CACHE.with(|cache| {
        *cache.borrow_mut() = None;
    });
}

fn current_engine_initialized() -> bool {
    CURRENT_ENGINE.with(|current| current.borrow().is_some())
}

fn set_last_engine_error(message: impl Into<String>) {
    LAST_ENGINE_ERROR.with(|last_error| {
        *last_error.borrow_mut() = message.into();
    });
}

fn clear_last_engine_error() {
    set_last_engine_error(String::new());
}

fn sync_start_request_error(engine: &BrowserEngine, request_id: u32) {
    if request_id == 0 {
        set_last_engine_error(engine.last_error().to_string());
    } else {
        clear_last_engine_error();
    }
}

fn cache_c_string(
    cache: &'static std::thread::LocalKey<RefCell<Option<CString>>>,
    value: String,
) -> *const c_char {
    cache.with(|cache| {
        let mut cache = cache.borrow_mut();
        *cache = Some(cstring_lossy(value));
        cache
            .as_ref()
            .map(|value| value.as_ptr())
            .unwrap_or(ptr::null())
    })
}

fn ensure_llama_cache_env() {
    if env::var_os("LLAMA_CACHE").is_some() {
        return;
    }
    let _ = fs::create_dir_all(LLAMA_CACHE_DIR);
    env::set_var("LLAMA_CACHE", LLAMA_CACHE_DIR);
}

fn enriched_backend_observability_json() -> String {
    let raw = backend_observability_json(true).unwrap_or_else(|_| "{}".to_string());
    let mut value = serde_json::from_str::<Value>(&raw)
        .ok()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    let Some(object) = value.as_object_mut() else {
        return "{}".to_string();
    };

    let compiled = object
        .get("compiled")
        .cloned()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    let webgpu_compiled = compiled
        .get("webgpu")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    object.entry("compiled").or_insert(compiled);
    object.insert("profilingEnabled".to_string(), Value::Bool(false));
    object.insert("webgpuCompiled".to_string(), Value::Bool(webgpu_compiled));

    let (webgpu_registered, webgpu_device_count) =
        webgpu_backend_status(object.get("availableBackends"));
    object.insert(
        "webgpuRegistered".to_string(),
        Value::Bool(webgpu_registered),
    );
    object.insert(
        "webgpuDeviceCount".to_string(),
        Value::from(webgpu_device_count),
    );
    object
        .entry("gpuOffloadSupported")
        .or_insert(Value::Bool(false));
    object
        .entry("availableBackends")
        .or_insert_with(|| Value::Array(Vec::new()));
    object
        .entry("devices")
        .or_insert_with(|| Value::Array(Vec::new()));
    object.insert(
        "engineInitialized".to_string(),
        Value::Bool(current_engine_initialized()),
    );

    value.to_string()
}

fn webgpu_backend_status(backends: Option<&Value>) -> (bool, u64) {
    let Some(backends) = backends.and_then(Value::as_array) else {
        return (false, 0);
    };
    for backend in backends {
        let Some(name) = backend.get("name").and_then(Value::as_str) else {
            continue;
        };
        if name.eq_ignore_ascii_case("WebGPU") {
            return (
                true,
                backend
                    .get("deviceCount")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
            );
        }
    }
    (false, 0)
}

fn is_valid_prediction_tokens(token_count: i32) -> bool {
    token_count > 0
}

fn read_size_arg(value: f64) -> Option<u64> {
    if value.is_finite() && value >= 0.0 && value <= MAX_EXACT_INTEGER {
        Some(value as u64)
    } else {
        None
    }
}

fn read_pointer_size_arg(value: f64) -> Option<usize> {
    let value = read_size_arg(value)?;
    usize::try_from(value).ok()
}

fn read_nonnegative_count(value: i32) -> Option<usize> {
    usize::try_from(value).ok()
}

unsafe fn required_cstr(value: *const c_char) -> Option<String> {
    if value.is_null() {
        return None;
    }
    Some(CStr::from_ptr(value).to_string_lossy().into_owned())
}

unsafe fn optional_cstr(value: *const c_char) -> String {
    required_cstr(value).unwrap_or_default()
}

unsafe fn bytes_from_raw<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        return Some(&[]);
    }
    if ptr.is_null() {
        return None;
    }
    Some(slice::from_raw_parts(ptr, len))
}

unsafe fn media_slices<'a>(
    images_flat_buffer: *const u8,
    image_sizes: *const i32,
    image_count: i32,
) -> Option<(&'a [u8], &'a [i32])> {
    let image_count = usize::try_from(image_count).ok()?;
    if image_count == 0 {
        return Some((&[], &[]));
    }
    if images_flat_buffer.is_null() || image_sizes.is_null() {
        return None;
    }
    let sizes = slice::from_raw_parts(image_sizes, image_count);
    let total_bytes = sizes.iter().try_fold(0usize, |sum, size| {
        let size = usize::try_from(*size).ok()?;
        sum.checked_add(size)
    })?;
    let images = bytes_from_raw(images_flat_buffer, total_bytes)?;
    Some((images, sizes))
}

unsafe fn mutable_u8_slice<'a>(ptr: *mut u8, len: i32) -> Option<&'a mut [u8]> {
    let len = read_nonnegative_count(len)?;
    if ptr.is_null() {
        return None;
    }
    Some(slice::from_raw_parts_mut(ptr, len))
}

fn byte_len_i32(bytes: &[u8]) -> i32 {
    i32::try_from(bytes.len()).unwrap_or(STATUS_FAILURE)
}

fn copy_bytes_with_nul(bytes: &[u8], buffer: &mut [u8]) -> i32 {
    if buffer.len() <= bytes.len() {
        return STATUS_INVALID_ARGUMENTS;
    }
    buffer[..bytes.len()].copy_from_slice(bytes);
    buffer[bytes.len()] = 0;
    byte_len_i32(bytes)
}

fn owned_json_error(code: &str, message: &str) -> *mut c_char {
    owned_string(json!({ "ok": false, "error": { "code": code, "message": message } }).to_string())
}

fn owned_string(value: String) -> *mut c_char {
    cstring_lossy(value).into_raw()
}

fn cstring_lossy(value: String) -> CString {
    let sanitized = value.replace('\0', "");
    match CString::new(sanitized) {
        Ok(value) => value,
        Err(_) => unsafe { CString::from_vec_unchecked(Vec::new()) },
    }
}
