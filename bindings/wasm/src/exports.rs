use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::{env, fs, ptr, slice};

use serde_json::{json, Value};
use sipp::backend::backend_observability_json;

use crate::engine::{BrowserEngine, BrowserMediaInput, BrowserTextRequestArgs, ABI_VERSION};
use crate::hash::BrowserSha256Hasher;
use crate::ingest::{
    GgufCloseShardCallback, GgufOpenShardCallback, GgufReadAtCallback, GgufWriteShardCallback,
};
use crate::{BrowserRuntimeMetrics, BrowserSchedulerLoopResult};

const STATUS_OK: i32 = 0;
const STATUS_FAILURE: i32 = -1;
const STATUS_INVALID_ARGUMENTS: i32 = -2;
const STATUS_STALE_RUNTIME_SESSION: i32 = -3;
const COMPLETED_REQUEST_STATUS_UNKNOWN: i32 = 4;
const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_991.0;
const LLAMA_CACHE_DIR: &str = "/tmp/sipp-llama-cache";

thread_local! {
    static CURRENT_ENGINE: RefCell<Option<Box<BrowserEngine>>> = const { RefCell::new(None) };
    static LAST_ENGINE_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
    static MEDIA_MARKER_CACHE: RefCell<Option<CString>> = const { RefCell::new(None) };
    static CHAT_TEMPLATE_CACHE: RefCell<Option<CString>> = const { RefCell::new(None) };
}

#[no_mangle]
pub extern "C" fn CE_RustBrowserEngineAbiVersion() -> i32 {
    ABI_VERSION as i32
}

#[no_mangle]
pub extern "C" fn CE_BrowserCacheLayout(
    source_bytes: f64,
    source_bytes_known: i32,
    direct_load_max_bytes: f64,
    shard_max_bytes: f64,
) -> i32 {
    catch_status(|| {
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
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_GgufPlanSplitCount(
    source_bytes: f64,
    shard_max_bytes: f64,
    user_data: *mut c_void,
    read_at: Option<GgufReadAtCallback>,
) -> i32 {
    catch_status(|| {
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
    })
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
    catch_status(|| {
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
            crate::ingest::GgufSplitCallbacks {
                user_data,
                read_at,
                open_shard,
                write_shard,
                close_shard,
            },
        )
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_DetectModelFromGgufBytes(
    name: *const c_char,
    bytes: *const u8,
    bytes_len: f64,
) -> *mut c_char {
    catch_owned_json("INVALID_GGUF", "GGUF model detection panicked", || {
        let Some(name) = required_cstr(name) else {
            return json_error_string("INVALID_GGUF", "invalid GGUF byte length");
        };
        let Some(bytes_len) = read_pointer_size_arg(bytes_len) else {
            return json_error_string("INVALID_GGUF", "invalid GGUF byte length");
        };
        let Some(bytes) = bytes_from_raw(bytes, bytes_len) else {
            return json_error_string("INVALID_GGUF", "invalid GGUF byte length");
        };
        crate::gguf::detect_model_from_gguf_bytes_json(&name, bytes)
    })
}

#[no_mangle]
pub extern "C" fn CE_Sha256Create() -> usize {
    catch_usize(|| Box::into_raw(Box::new(BrowserSha256Hasher::new())) as usize)
}

#[no_mangle]
pub unsafe extern "C" fn CE_Sha256Update(hasher: usize, bytes: *const u8, bytes_len: f64) -> i32 {
    catch_status(|| {
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
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_Sha256Finalize(hasher: usize) -> *mut c_char {
    catch_ptr(|| {
        if hasher == 0 {
            return ptr::null_mut();
        }
        let hasher = Box::from_raw(hasher as *mut BrowserSha256Hasher);
        owned_string(hasher.finalize_hex())
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_Sha256Close(hasher: usize) -> i32 {
    catch_status(|| {
        if hasher == 0 {
            return STATUS_INVALID_ARGUMENTS;
        }
        drop(Box::from_raw(hasher as *mut BrowserSha256Hasher));
        1
    })
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

/// Installs a browser model from serialized managed assets.
///
/// # Safety
///
/// `source_json` must be null or point to a valid NUL-terminated string for
/// the duration of this call. The returned string must be freed with
/// [`CE_FreeString`].
#[no_mangle]
pub unsafe extern "C" fn CE_ModelServiceInstall(
    service: usize,
    source_json: *const c_char,
) -> *mut c_char {
    let Some(source_json) = required_cstr(source_json) else {
        return owned_json_error("INVALID_MODEL_SOURCE", "install source JSON is missing");
    };
    owned_string(crate::lifecycle::model_service_install_json(
        service,
        &source_json,
    ))
}

/// Advances the browser remote-acquisition protocol through an owned JSON string.
///
/// # Safety
///
/// `command_json` must be null or point to a valid NUL-terminated string for
/// the duration of this call. The returned string must be released with
/// [`CE_FreeString`].
#[no_mangle]
pub unsafe extern "C" fn CE_ModelServiceRemoteAcquisitionCommand(
    service: usize,
    command_json: *const c_char,
) -> *mut c_char {
    let Some(command_json) = required_cstr(command_json) else {
        return owned_json_error(
            "INVALID_MODEL_SOURCE",
            "remote acquisition command JSON is missing",
        );
    };
    owned_string(
        crate::lifecycle::model_service_remote_acquisition_command_json(service, &command_json),
    )
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
pub unsafe extern "C" fn CE_ModelServiceRemove(
    service: usize,
    request_json: *const c_char,
) -> *mut c_char {
    let Some(request_json) = required_cstr(request_json) else {
        return owned_json_error("INVALID_MODEL_SOURCE", "remove request JSON is missing");
    };
    owned_string(crate::lifecycle::model_service_remove_json(
        service,
        &request_json,
    ))
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
    session_descriptor_json: *const c_char,
) -> i32 {
    let (Some(model_path), Some(runtime_config_json), Some(session_descriptor_json)) = (
        required_cstr(model_path),
        required_cstr(runtime_config_json),
        required_cstr(session_descriptor_json),
    ) else {
        set_last_engine_error("engine init received a null string");
        return STATUS_INVALID_ARGUMENTS;
    };

    let activation = match BrowserEngine::prepare_load(
        &model_path,
        &runtime_config_json,
        &session_descriptor_json,
    ) {
        Ok(activation) => activation,
        Err(error) => {
            set_last_engine_error(error);
            return STATUS_INVALID_ARGUMENTS;
        }
    };

    close_current_engine();
    ensure_llama_cache_env();

    let mut engine = Box::new(BrowserEngine::create());
    let status = engine.load(activation);
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
pub extern "C" fn CE_GetRuntimeSessionJson() -> *mut c_char {
    let session = with_current_engine(None, |engine| Some(engine.runtime_session_json()));
    match session {
        Some(Ok(session)) => owned_string(session),
        Some(Err(error)) => {
            set_last_engine_error(error);
            ptr::null_mut()
        }
        None => {
            set_last_engine_error("browser runtime session is not loaded");
            ptr::null_mut()
        }
    }
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
pub unsafe extern "C" fn CE_PairingValidate(classified_json: *const c_char) -> *mut c_char {
    let Some(classified_json) = required_cstr(classified_json) else {
        return owned_json_error("INVALID_MODEL_SOURCE", "classified asset JSON is missing");
    };
    owned_string(crate::pairing::pairing_validate_json(&classified_json))
}

#[no_mangle]
pub extern "C" fn CE_ProbeChatBoundaryInfo() -> *mut c_char {
    owned_string(with_current_engine(
        String::new(),
        BrowserEngine::probe_chat_boundary_info,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartTextRequest(
    generation: u32,
    context_key: *const c_char,
    prompt: *const c_char,
    n_tokens: i32,
    token_emission_mode: i32,
    grammar: *const c_char,
    stop_json: *const c_char,
    sampling_json: *const c_char,
) -> u32 {
    if prompt.is_null() || !is_valid_prediction_tokens(n_tokens) {
        return 0;
    }
    let context_key = optional_cstr(context_key);
    let prompt = optional_cstr(prompt);
    let grammar = optional_cstr(grammar);
    let stop_json = optional_cstr(stop_json);
    let sampling_json = optional_cstr(sampling_json);
    with_session_engine_mut(generation, 0, |engine| {
        let request_id = engine.start_text_request(
            &context_key,
            &prompt,
            n_tokens,
            BrowserTextRequestArgs {
                emit_tokens: token_emission_mode,
                grammar: &grammar,
                stop_json: &stop_json,
                sampling_json: &sampling_json,
            },
        );
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartMediaRequest(
    generation: u32,
    context_key: *const c_char,
    prompt: *const c_char,
    n_tokens: i32,
    n_images: i32,
    images_flat_buffer: *const u8,
    image_sizes: *const i32,
    token_emission_mode: i32,
    grammar: *const c_char,
    stop_json: *const c_char,
    sampling_json: *const c_char,
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
    let stop_json = optional_cstr(stop_json);
    let sampling_json = optional_cstr(sampling_json);
    with_session_engine_mut(generation, 0, |engine| {
        let request_id = engine.start_media_request(
            &context_key,
            &prompt,
            n_tokens,
            BrowserMediaInput {
                flat_buffer: images,
                sizes,
            },
            BrowserTextRequestArgs {
                emit_tokens: token_emission_mode,
                grammar: &grammar,
                stop_json: &stop_json,
                sampling_json: &sampling_json,
            },
        );
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartChatRequest(
    generation: u32,
    context_key: *const c_char,
    messages_json: *const c_char,
    n_tokens: i32,
    n_images: i32,
    images_flat_buffer: *const u8,
    image_sizes: *const i32,
    token_emission_mode: i32,
    grammar: *const c_char,
    stop_json: *const c_char,
    sampling_json: *const c_char,
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
    let stop_json = optional_cstr(stop_json);
    let sampling_json = optional_cstr(sampling_json);
    with_session_engine_mut(generation, 0, |engine| {
        let request_id = engine.start_chat_request(
            &context_key,
            &messages_json,
            n_tokens,
            BrowserMediaInput {
                flat_buffer: images,
                sizes,
            },
            BrowserTextRequestArgs {
                emit_tokens: token_emission_mode,
                grammar: &grammar,
                stop_json: &stop_json,
                sampling_json: &sampling_json,
            },
        );
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartEmbeddingRequest(
    generation: u32,
    context_key: *const c_char,
    input: *const c_char,
    normalize: i32,
) -> u32 {
    if input.is_null() {
        return 0;
    }
    let context_key = optional_cstr(context_key);
    let input = optional_cstr(input);
    with_session_engine_mut(generation, 0, |engine| {
        let request_id = engine.start_embedding_request(&context_key, &input, normalize);
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartListenRequest(
    generation: u32,
    audio: *const u8,
    audio_len: i32,
    language: *const c_char,
    max_tokens: i32,
) -> u32 {
    let Some(audio_len) = read_nonnegative_count(audio_len) else {
        return 0;
    };
    let Some(audio) = bytes_from_raw(audio, audio_len) else {
        set_last_engine_error("listen audio buffer is invalid");
        return 0;
    };
    let language = optional_cstr(language);
    with_session_engine_mut(generation, 0, |engine| {
        let request_id = engine.start_listen_request(audio, &language, max_tokens);
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_StartSpeakRequest(
    generation: u32,
    text: *const c_char,
    language: *const c_char,
    speaker_audio: *const u8,
    speaker_audio_len: i32,
    has_max_duration: i32,
    max_duration_ms: u32,
) -> u32 {
    let Some(text) = required_cstr(text) else {
        return 0;
    };
    let Some(speaker_audio_len) = read_nonnegative_count(speaker_audio_len) else {
        return 0;
    };
    let Some(speaker_audio) = bytes_from_raw(speaker_audio, speaker_audio_len) else {
        set_last_engine_error("speaker audio buffer is invalid");
        return 0;
    };
    let max_duration_ms = match has_max_duration {
        0 => None,
        1 => Some(max_duration_ms),
        _ => {
            set_last_engine_error("has_max_duration must be 0 or 1");
            return 0;
        }
    };
    let language = optional_cstr(language);
    with_session_engine_mut(generation, 0, |engine| {
        let request_id =
            engine.start_speak_request(&text, &language, speaker_audio, max_duration_ms);
        sync_start_request_error(engine, request_id);
        request_id
    })
}

#[no_mangle]
pub extern "C" fn CE_CancelRequest(generation: u32, request_id: u32) -> i32 {
    if request_id == 0 {
        return 0;
    }
    with_session_engine_mut(generation, STATUS_STALE_RUNTIME_SESSION, |engine| {
        engine.cancel_request(request_id)
    })
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
    generation: u32,
    max_ticks: i32,
    max_completed_responses: i32,
    max_generated_tokens: i32,
    out_result: *mut BrowserSchedulerLoopResult,
) -> i32 {
    if out_result.is_null() {
        return STATUS_FAILURE;
    }
    with_session_engine_mut(generation, STATUS_STALE_RUNTIME_SESSION, |engine| {
        engine.run_scheduler_loop(
            max_ticks,
            max_completed_responses,
            max_generated_tokens,
            &mut *out_result,
        )
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestStatus(generation: u32, request_id: u32) -> i32 {
    if request_id == 0 {
        return COMPLETED_REQUEST_STATUS_UNKNOWN;
    }
    with_session_engine(generation, STATUS_STALE_RUNTIME_SESSION, |engine| {
        engine.completed_status(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestOutputKind(generation: u32, request_id: u32) -> i32 {
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_output_kind(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetTokenRingHeaderAddress() -> *const u32 {
    with_current_engine(ptr::null(), |engine| {
        engine.token_ring_header_address() as *const u32
    })
}

#[no_mangle]
pub extern "C" fn CE_GetTokenRingBodyAddress() -> *const u8 {
    with_current_engine(ptr::null(), |engine| {
        engine.token_ring_body_address() as *const u8
    })
}

#[no_mangle]
pub extern "C" fn CE_GetTokenRingCapacity() -> i32 {
    with_current_engine(0, BrowserEngine::token_ring_capacity)
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestOutputSize(generation: u32, request_id: u32) -> i32 {
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_output_size(request_id)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_CopyCompletedRequestOutput(
    generation: u32,
    request_id: u32,
    buffer: *mut c_char,
    capacity: i32,
) -> i32 {
    let Some(buffer) = mutable_u8_slice(buffer as *mut u8, capacity) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.copy_completed_output(request_id, buffer)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestEmbeddingLength(generation: u32, request_id: u32) -> i32 {
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_embedding_len(request_id)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_CopyCompletedRequestEmbedding(
    generation: u32,
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
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.copy_completed_embedding(request_id, buffer)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestEmbeddingPooling(generation: u32, request_id: u32) -> i32 {
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_embedding_pooling(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestEmbeddingNormalized(
    generation: u32,
    request_id: u32,
) -> i32 {
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_embedding_normalized(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestAudioLength(generation: u32, request_id: u32) -> i32 {
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_audio_len(request_id)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_CopyCompletedRequestAudio(
    generation: u32,
    request_id: u32,
    buffer: *mut u8,
    capacity: i32,
) -> i32 {
    let Some(buffer) = mutable_u8_slice(buffer, capacity) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.copy_completed_audio(request_id, buffer)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestAudioSampleRate(generation: u32, request_id: u32) -> i32 {
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_audio_sample_rate(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestAudioChannels(generation: u32, request_id: u32) -> i32 {
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_audio_channels(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestAudioDurationMs(generation: u32, request_id: u32) -> f64 {
    with_session_engine(generation, f64::from(STATUS_FAILURE), |engine| {
        engine.completed_audio_duration_ms(request_id)
    })
}

#[no_mangle]
pub extern "C" fn CE_GetCompletedRequestErrorSize(generation: u32, request_id: u32) -> i32 {
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_error_size(request_id)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_CopyCompletedRequestError(
    generation: u32,
    request_id: u32,
    buffer: *mut c_char,
    capacity: i32,
) -> i32 {
    let Some(buffer) = mutable_u8_slice(buffer as *mut u8, capacity) else {
        return STATUS_INVALID_ARGUMENTS;
    };
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.copy_completed_error(request_id, buffer)
    })
}

#[no_mangle]
pub unsafe extern "C" fn CE_GetCompletedRequestRuntimeObservability(
    generation: u32,
    request_id: u32,
    out_metrics: *mut BrowserRuntimeMetrics,
) -> i32 {
    if out_metrics.is_null() {
        return STATUS_FAILURE;
    }
    with_session_engine(generation, STATUS_FAILURE, |engine| {
        engine.completed_runtime_observability(request_id, &mut *out_metrics)
    })
}

#[no_mangle]
pub extern "C" fn CE_ConsumeCompletedRequest(generation: u32, request_id: u32) -> i32 {
    with_session_engine_mut(generation, STATUS_STALE_RUNTIME_SESSION, |engine| {
        engine.consume_completed_request(request_id)
    })
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

fn with_session_engine<T>(
    generation: u32,
    failure: T,
    operation: impl FnOnce(&BrowserEngine) -> T,
) -> T {
    CURRENT_ENGINE.with(|current| {
        let current = current.borrow();
        let Some(engine) = current.as_deref() else {
            set_last_engine_error("browser runtime session is not loaded");
            return failure;
        };
        if engine.id() != generation {
            set_last_engine_error(format!(
                "browser runtime generation {generation} is stale; current generation is {}",
                engine.id()
            ));
            return failure;
        }
        operation(engine)
    })
}

fn with_session_engine_mut<T>(
    generation: u32,
    failure: T,
    operation: impl FnOnce(&mut BrowserEngine) -> T,
) -> T {
    CURRENT_ENGINE.with(|current| {
        let Some(mut engine) = current.borrow_mut().take() else {
            set_last_engine_error("browser runtime session is not loaded");
            return failure;
        };
        if engine.id() != generation {
            set_last_engine_error(format!(
                "browser runtime generation {generation} is stale; current generation is {}",
                engine.id()
            ));
            *current.borrow_mut() = Some(engine);
            return failure;
        }
        let result = operation(&mut engine);
        *current.borrow_mut() = Some(engine);
        result
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

#[cfg(test)]
#[path = "tests/exports_tests.rs"]
mod exports_tests;

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
    if value.is_finite() && (0.0..=MAX_EXACT_INTEGER).contains(&value) {
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

fn catch_status(operation: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or(STATUS_FAILURE)
}

fn catch_usize(operation: impl FnOnce() -> usize) -> usize {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or(0)
}

fn catch_ptr(operation: impl FnOnce() -> *mut c_char) -> *mut c_char {
    catch_unwind(AssertUnwindSafe(operation)).unwrap_or(ptr::null_mut())
}

fn catch_owned_json(
    code: &str,
    panic_message: &str,
    operation: impl FnOnce() -> String,
) -> *mut c_char {
    let response = catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or_else(|_| json_error_string(code, panic_message));
    owned_string(response)
}

fn owned_json_error(code: &str, message: &str) -> *mut c_char {
    owned_string(json_error_string(code, message))
}

fn json_error_string(code: &str, message: &str) -> String {
    json!({ "ok": false, "error": { "code": code, "message": message } }).to_string()
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
