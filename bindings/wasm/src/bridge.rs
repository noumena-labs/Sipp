use crate::engine::BrowserEngine;
use crate::hash::BrowserSha256Hasher;

#[cxx::bridge(namespace = "cogentlm::wasm")]
pub mod ffi {
    struct BrowserSchedulerLoopResult {
        ticks_executed: i32,
        progressed_ticks: i32,
        completed_response_count: i32,
        emitted_token_count: i32,
    }

    struct BrowserRuntimeMetrics {
        ttft_ms: f64,
        itl_avg_ms: f64,
        itl_p99_ms: f64,
        e2e_ms: f64,
        prefill_ms: f64,
        decode_ms: f64,
        native_gpu_ms: f64,
        native_sync_ms: f64,
        native_logic_ms: f64,
        input_tokens: i32,
        output_tokens: i32,
        cache_hits: i32,
        prefill_tokens: i32,
    }

    unsafe extern "C++" {
        include!("gguf_callbacks.h");

        type GgufReadAt;
        type GgufShardSink;
        type GgufShardWriter;

        fn read_at(self: Pin<&mut GgufReadAt>, offset: u64, dst: &mut [u8]) -> i32;
        fn open_shard(self: Pin<&mut GgufShardSink>, path: &str, index: u16, count: u16) -> i32;
        fn create_writer(self: Pin<&mut GgufShardSink>) -> UniquePtr<GgufShardWriter>;
        fn write_shard(self: Pin<&mut GgufShardWriter>, bytes: &[u8]) -> i32;
        fn close_shard(self: Pin<&mut GgufShardWriter>) -> i32;
    }

    extern "Rust" {
        type BrowserEngine;
        type BrowserSha256Hasher;

        fn browser_engine_abi_version() -> u32;
        fn browser_engine_create() -> Box<BrowserEngine>;
        fn browser_engine_id(engine: &BrowserEngine) -> u32;
        fn browser_engine_load(
            engine: &mut BrowserEngine,
            model_path: &str,
            runtime_config_json: &str,
        ) -> i32;
        fn browser_engine_last_error_size(engine: &BrowserEngine) -> i32;
        fn browser_engine_copy_last_error(engine: &BrowserEngine, buffer: &mut [u8]) -> i32;
        fn browser_engine_start_text_request(
            engine: &mut BrowserEngine,
            context_key: &str,
            prompt: &str,
            max_tokens: i32,
            token_emission_mode: i32,
            grammar: &str,
        ) -> u32;
        fn browser_engine_start_media_request(
            engine: &mut BrowserEngine,
            context_key: &str,
            prompt: &str,
            max_tokens: i32,
            images_flat_buffer: &[u8],
            image_sizes: &[i32],
            token_emission_mode: i32,
            grammar: &str,
        ) -> u32;
        fn browser_engine_start_chat_request(
            engine: &mut BrowserEngine,
            context_key: &str,
            messages_json: &str,
            max_tokens: i32,
            images_flat_buffer: &[u8],
            image_sizes: &[i32],
            token_emission_mode: i32,
            grammar: &str,
        ) -> u32;
        fn browser_engine_start_embedding_request(
            engine: &mut BrowserEngine,
            context_key: &str,
            input: &str,
            normalize: i32,
        ) -> u32;
        fn browser_engine_cancel_request(engine: &mut BrowserEngine, request_id: u32) -> i32;
        fn browser_engine_run_scheduler_loop(
            engine: &mut BrowserEngine,
            max_ticks: i32,
            max_completed_responses: i32,
            max_emitted_tokens: i32,
            max_duration_us: i32,
            streaming_active: bool,
            out: &mut BrowserSchedulerLoopResult,
        ) -> i32;
        fn browser_engine_completed_request_status(engine: &BrowserEngine, request_id: u32) -> i32;
        fn browser_engine_completed_request_output_kind(
            engine: &BrowserEngine,
            request_id: u32,
        ) -> i32;
        fn browser_engine_completed_request_output_size(
            engine: &BrowserEngine,
            request_id: u32,
        ) -> i32;
        fn browser_engine_copy_completed_request_output(
            engine: &BrowserEngine,
            request_id: u32,
            buffer: &mut [u8],
        ) -> i32;
        fn browser_engine_completed_request_embedding_length(
            engine: &BrowserEngine,
            request_id: u32,
        ) -> i32;
        fn browser_engine_copy_completed_request_embedding(
            engine: &BrowserEngine,
            request_id: u32,
            buffer: &mut [f32],
        ) -> i32;
        fn browser_engine_completed_request_embedding_pooling(
            engine: &BrowserEngine,
            request_id: u32,
        ) -> i32;
        fn browser_engine_completed_request_embedding_normalized(
            engine: &BrowserEngine,
            request_id: u32,
        ) -> i32;
        fn browser_engine_completed_request_error_size(
            engine: &BrowserEngine,
            request_id: u32,
        ) -> i32;
        fn browser_engine_copy_completed_request_error(
            engine: &BrowserEngine,
            request_id: u32,
            buffer: &mut [u8],
        ) -> i32;
        fn browser_engine_consume_completed_request(
            engine: &mut BrowserEngine,
            request_id: u32,
        ) -> i32;
        fn browser_engine_runtime_observability(
            engine: &BrowserEngine,
            out: &mut BrowserRuntimeMetrics,
        ) -> i32;
        fn browser_engine_completed_runtime_observability(
            engine: &BrowserEngine,
            request_id: u32,
            out: &mut BrowserRuntimeMetrics,
        ) -> i32;
        fn browser_engine_streaming_buffer_pointer(engine: &mut BrowserEngine) -> usize;
        fn browser_engine_streaming_buffer_used_address(engine: &mut BrowserEngine) -> usize;
        fn browser_engine_streaming_buffer_drop_count_address(engine: &mut BrowserEngine) -> usize;
        fn browser_engine_media_marker(engine: &BrowserEngine) -> String;
        fn browser_engine_chat_template(engine: &BrowserEngine) -> String;
        fn browser_engine_bos_text(engine: &BrowserEngine) -> String;
        fn browser_engine_eos_text(engine: &BrowserEngine) -> String;
        fn browser_engine_probe_chat_boundary_info(engine: &BrowserEngine) -> String;

        fn browser_cache_layout(
            source_bytes: u64,
            source_bytes_known: bool,
            direct_load_max_bytes: u64,
            shard_max_bytes: u64,
        ) -> i32;
        fn detect_model_from_gguf_bytes_json(name: &str, bytes: &[u8]) -> String;
        fn gguf_plan_split_count(
            source_bytes: u64,
            shard_max_bytes: u64,
            source: Pin<&mut GgufReadAt>,
        ) -> i32;
        fn gguf_split_stream(
            source_bytes: u64,
            output_prefix: &str,
            shard_max_bytes: u64,
            source: Pin<&mut GgufReadAt>,
            sink: Pin<&mut GgufShardSink>,
        ) -> i32;

        fn sha256_create() -> Box<BrowserSha256Hasher>;
        fn sha256_update(hasher: &mut BrowserSha256Hasher, bytes: &[u8]);
        fn sha256_finalize(hasher: Box<BrowserSha256Hasher>) -> String;

        #[cfg(target_family = "wasm")]
        fn pairing_validate_json(classified_json: &str, explicit_projector_id: &str) -> String;

        #[cfg(target_family = "wasm")]
        fn model_service_create_json(config_json: &str) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_close(service: usize) -> i32;
        #[cfg(target_family = "wasm")]
        fn model_service_list_json(service: usize) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_current_json(service: usize) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_manifest_json(service: usize) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_prepare_load_json(
            service: usize,
            source_json: &str,
            options_json: &str,
        ) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_commit_load_json(service: usize, commit_json: &str) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_abort_load_json(service: usize, error_json: &str) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_remove_json(service: usize, model_id: &str) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_unload_json(service: usize) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_snapshot_json(service: usize) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_drain_events_json(service: usize) -> String;
        #[cfg(target_family = "wasm")]
        fn model_service_record_event_json(
            service: usize,
            event_type: &str,
            patch_json: &str,
        ) -> String;
    }
}

pub use ffi::{BrowserRuntimeMetrics, BrowserSchedulerLoopResult};

fn browser_engine_abi_version() -> u32 {
    crate::engine::ABI_VERSION
}

fn browser_engine_create() -> Box<BrowserEngine> {
    Box::new(BrowserEngine::create())
}

fn browser_engine_id(engine: &BrowserEngine) -> u32 {
    engine.id()
}

fn browser_engine_load(
    engine: &mut BrowserEngine,
    model_path: &str,
    runtime_config_json: &str,
) -> i32 {
    engine.load(model_path, runtime_config_json)
}

fn browser_engine_last_error_size(engine: &BrowserEngine) -> i32 {
    engine.last_error_size()
}

fn browser_engine_copy_last_error(engine: &BrowserEngine, buffer: &mut [u8]) -> i32 {
    engine.copy_last_error(buffer)
}

fn browser_engine_start_text_request(
    engine: &mut BrowserEngine,
    context_key: &str,
    prompt: &str,
    max_tokens: i32,
    token_emission_mode: i32,
    grammar: &str,
) -> u32 {
    engine.start_text_request(
        context_key,
        prompt,
        max_tokens,
        token_emission_mode,
        grammar,
    )
}

fn browser_engine_start_media_request(
    engine: &mut BrowserEngine,
    context_key: &str,
    prompt: &str,
    max_tokens: i32,
    images_flat_buffer: &[u8],
    image_sizes: &[i32],
    token_emission_mode: i32,
    grammar: &str,
) -> u32 {
    engine.start_media_request(
        context_key,
        prompt,
        max_tokens,
        images_flat_buffer,
        image_sizes,
        token_emission_mode,
        grammar,
    )
}

fn browser_engine_start_chat_request(
    engine: &mut BrowserEngine,
    context_key: &str,
    messages_json: &str,
    max_tokens: i32,
    images_flat_buffer: &[u8],
    image_sizes: &[i32],
    token_emission_mode: i32,
    grammar: &str,
) -> u32 {
    engine.start_chat_request(
        context_key,
        messages_json,
        max_tokens,
        images_flat_buffer,
        image_sizes,
        token_emission_mode,
        grammar,
    )
}

fn browser_engine_start_embedding_request(
    engine: &mut BrowserEngine,
    context_key: &str,
    input: &str,
    normalize: i32,
) -> u32 {
    engine.start_embedding_request(context_key, input, normalize)
}

fn browser_engine_cancel_request(engine: &mut BrowserEngine, request_id: u32) -> i32 {
    engine.cancel_request(request_id)
}

fn browser_engine_run_scheduler_loop(
    engine: &mut BrowserEngine,
    max_ticks: i32,
    max_completed_responses: i32,
    max_emitted_tokens: i32,
    max_duration_us: i32,
    streaming_active: bool,
    out: &mut BrowserSchedulerLoopResult,
) -> i32 {
    engine.run_scheduler_loop(
        max_ticks,
        max_completed_responses,
        max_emitted_tokens,
        max_duration_us,
        streaming_active,
        out,
    )
}

fn browser_engine_completed_request_status(engine: &BrowserEngine, request_id: u32) -> i32 {
    engine.completed_status(request_id)
}

fn browser_engine_completed_request_output_kind(engine: &BrowserEngine, request_id: u32) -> i32 {
    engine.completed_output_kind(request_id)
}

fn browser_engine_completed_request_output_size(engine: &BrowserEngine, request_id: u32) -> i32 {
    engine.completed_output_size(request_id)
}

fn browser_engine_copy_completed_request_output(
    engine: &BrowserEngine,
    request_id: u32,
    buffer: &mut [u8],
) -> i32 {
    engine.copy_completed_output(request_id, buffer)
}

fn browser_engine_completed_request_embedding_length(
    engine: &BrowserEngine,
    request_id: u32,
) -> i32 {
    engine.completed_embedding_len(request_id)
}

fn browser_engine_copy_completed_request_embedding(
    engine: &BrowserEngine,
    request_id: u32,
    buffer: &mut [f32],
) -> i32 {
    engine.copy_completed_embedding(request_id, buffer)
}

fn browser_engine_completed_request_embedding_pooling(
    engine: &BrowserEngine,
    request_id: u32,
) -> i32 {
    engine.completed_embedding_pooling(request_id)
}

fn browser_engine_completed_request_embedding_normalized(
    engine: &BrowserEngine,
    request_id: u32,
) -> i32 {
    engine.completed_embedding_normalized(request_id)
}

fn browser_engine_completed_request_error_size(engine: &BrowserEngine, request_id: u32) -> i32 {
    engine.completed_error_size(request_id)
}

fn browser_engine_copy_completed_request_error(
    engine: &BrowserEngine,
    request_id: u32,
    buffer: &mut [u8],
) -> i32 {
    engine.copy_completed_error(request_id, buffer)
}

fn browser_engine_consume_completed_request(engine: &mut BrowserEngine, request_id: u32) -> i32 {
    engine.consume_completed_request(request_id)
}

fn browser_engine_runtime_observability(
    engine: &BrowserEngine,
    out: &mut BrowserRuntimeMetrics,
) -> i32 {
    engine.runtime_observability(out)
}

fn browser_engine_completed_runtime_observability(
    engine: &BrowserEngine,
    request_id: u32,
    out: &mut BrowserRuntimeMetrics,
) -> i32 {
    engine.completed_runtime_observability(request_id, out)
}

fn browser_engine_streaming_buffer_pointer(engine: &mut BrowserEngine) -> usize {
    engine.streaming_buffer_ptr()
}

fn browser_engine_streaming_buffer_used_address(engine: &mut BrowserEngine) -> usize {
    engine.streaming_buffer_used_address()
}

fn browser_engine_streaming_buffer_drop_count_address(engine: &mut BrowserEngine) -> usize {
    engine.streaming_buffer_drop_count_address()
}

fn browser_engine_media_marker(engine: &BrowserEngine) -> String {
    engine.media_marker()
}

fn browser_engine_chat_template(engine: &BrowserEngine) -> String {
    engine.chat_template_source()
}

fn browser_engine_bos_text(engine: &BrowserEngine) -> String {
    engine.bos_text()
}

fn browser_engine_eos_text(engine: &BrowserEngine) -> String {
    engine.eos_text()
}

fn browser_engine_probe_chat_boundary_info(engine: &BrowserEngine) -> String {
    engine.probe_chat_boundary_info()
}

fn browser_cache_layout(
    source_bytes: u64,
    source_bytes_known: bool,
    direct_load_max_bytes: u64,
    shard_max_bytes: u64,
) -> i32 {
    crate::ingest::browser_cache_layout(
        source_bytes,
        source_bytes_known,
        direct_load_max_bytes,
        shard_max_bytes,
    )
}

fn detect_model_from_gguf_bytes_json(name: &str, bytes: &[u8]) -> String {
    crate::gguf::detect_model_from_gguf_bytes_json(name, bytes)
}

fn gguf_plan_split_count(
    source_bytes: u64,
    shard_max_bytes: u64,
    source: std::pin::Pin<&mut ffi::GgufReadAt>,
) -> i32 {
    crate::ingest::gguf_plan_split_count(source_bytes, shard_max_bytes, source)
}

fn gguf_split_stream(
    source_bytes: u64,
    output_prefix: &str,
    shard_max_bytes: u64,
    source: std::pin::Pin<&mut ffi::GgufReadAt>,
    sink: std::pin::Pin<&mut ffi::GgufShardSink>,
) -> i32 {
    crate::ingest::gguf_split_stream(source_bytes, output_prefix, shard_max_bytes, source, sink)
}

fn sha256_create() -> Box<BrowserSha256Hasher> {
    Box::new(BrowserSha256Hasher::new())
}

fn sha256_update(hasher: &mut BrowserSha256Hasher, bytes: &[u8]) {
    hasher.update(bytes);
}

fn sha256_finalize(hasher: Box<BrowserSha256Hasher>) -> String {
    let hasher = *hasher;
    hasher.finalize_hex()
}

#[cfg(target_family = "wasm")]
fn pairing_validate_json(classified_json: &str, explicit_projector_id: &str) -> String {
    crate::pairing::pairing_validate_json(classified_json, explicit_projector_id)
}

#[cfg(target_family = "wasm")]
fn model_service_create_json(config_json: &str) -> String {
    crate::lifecycle::model_service_create_json(config_json)
}

#[cfg(target_family = "wasm")]
fn model_service_close(service: usize) -> i32 {
    crate::lifecycle::model_service_close(service)
}

#[cfg(target_family = "wasm")]
fn model_service_list_json(service: usize) -> String {
    crate::lifecycle::model_service_list_json(service)
}

#[cfg(target_family = "wasm")]
fn model_service_current_json(service: usize) -> String {
    crate::lifecycle::model_service_current_json(service)
}

#[cfg(target_family = "wasm")]
fn model_service_manifest_json(service: usize) -> String {
    crate::lifecycle::model_service_manifest_json(service)
}

#[cfg(target_family = "wasm")]
fn model_service_prepare_load_json(
    service: usize,
    source_json: &str,
    options_json: &str,
) -> String {
    crate::lifecycle::model_service_prepare_load_json(service, source_json, options_json)
}

#[cfg(target_family = "wasm")]
fn model_service_commit_load_json(service: usize, commit_json: &str) -> String {
    crate::lifecycle::model_service_commit_load_json(service, commit_json)
}

#[cfg(target_family = "wasm")]
fn model_service_abort_load_json(service: usize, error_json: &str) -> String {
    crate::lifecycle::model_service_abort_load_json(service, error_json)
}

#[cfg(target_family = "wasm")]
fn model_service_remove_json(service: usize, model_id: &str) -> String {
    crate::lifecycle::model_service_remove_json(service, model_id)
}

#[cfg(target_family = "wasm")]
fn model_service_unload_json(service: usize) -> String {
    crate::lifecycle::model_service_unload_json(service)
}

#[cfg(target_family = "wasm")]
fn model_service_snapshot_json(service: usize) -> String {
    crate::lifecycle::model_service_snapshot_json(service)
}

#[cfg(target_family = "wasm")]
fn model_service_drain_events_json(service: usize) -> String {
    crate::lifecycle::model_service_drain_events_json(service)
}

#[cfg(target_family = "wasm")]
fn model_service_record_event_json(service: usize, event_type: &str, patch_json: &str) -> String {
    crate::lifecycle::model_service_record_event_json(service, event_type, patch_json)
}
