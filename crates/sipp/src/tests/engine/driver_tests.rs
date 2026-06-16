//! Tests the `engine::driver` module in `sipp`.
//!
//! Covers driver futures, command handling, event emission, and request mapping with model-free channels or explicitly ignored model smoke tests.

use super::*;
use crate::core::{TokenBatch, TokenEmissionStats};
use crate::engine::{
    CacheSource, EmbedOptions, GenerateOptions, GpuLayerConfig, KvReuseMode, NativeRuntimeConfig,
    SamplingRuntimeOverride, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};
use crate::runtime::request::GenerateResponse;
use futures::executor::block_on;
use futures::future::poll_fn;
use futures::StreamExt;
use std::path::PathBuf;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

fn repo_fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join(name)
}

fn cache_smoke_config(mode: KvReuseMode) -> NativeRuntimeConfig {
    let mut config = NativeRuntimeConfig::default();
    config.placement.gpu_layers = GpuLayerConfig::Count(0);
    config.context.n_ctx = Some(256);
    config.context.n_batch = Some(64);
    config.context.n_ubatch = Some(64);
    config.context.n_threads = Some(2);
    config.context.n_threads_batch = Some(2);
    config.context.warmup = false;
    config.cache.mode = mode;
    config.observability.runtime_metrics = true;
    config
}

fn cache_query(context_key: &str, prompt: &str) -> QueryRequest {
    QueryRequest::new(prompt).options(QueryOptions {
        context_key: context_key.to_string(),
        max_tokens: 1,
        ..QueryOptions::default()
    })
}

fn token_batch(text: &str) -> TokenBatch {
    TokenBatch {
        request_id: "req".to_string(),
        stream_id: 1,
        sequence_start: 0,
        text: text.to_string(),
        frame_count: 1,
        byte_count: text.len() as u32,
        stats: TokenEmissionStats {
            frames_sent: 1,
            bytes_sent: text.len() as u64,
            batches_sent: 1,
        },
    }
}

fn closed_engine() -> SippEngine {
    let (command_tx, command_rx) = mpsc::channel();
    drop(command_rx);
    SippEngine {
        inner: Arc::new(EngineInner {
            command_tx,
            event_subscribers: Arc::new(Mutex::new(Vec::new())),
            _driver: thread::spawn(|| {}),
        }),
    }
}

#[test]
fn query_options_default_matches_public_completion_defaults() {
    let options = QueryOptions::default();

    assert_eq!(options.context_key, DEFAULT_CONTEXT_KEY);
    assert_eq!(options.max_tokens, DEFAULT_MAX_TOKENS);
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
        sampling: Some(SamplingRuntimeOverride {
            temperature: Some(0.1),
            ..SamplingRuntimeOverride::default()
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
    let sampling = options.sampling.as_ref().expect("sampling override");
    assert_eq!(sampling.temperature, Some(0.1));
}

#[test]
fn generate_options_without_cache_key_uses_default_context() {
    let options = QueryOptions::from(GenerateOptions {
        cache_key: None,
        ..GenerateOptions::default()
    });

    assert_eq!(options.context_key, DEFAULT_CONTEXT_KEY);
}

#[test]
fn query_request_defaults_options() {
    let request = QueryRequest::new("hello");

    assert_eq!(request.prompt, "hello");
    assert_eq!(request.options, QueryOptions::default());
}

#[test]
fn engine_handle_is_send() {
    fn assert_send<T: Send>() {}
    assert_send::<SippEngine>();
}

#[test]
fn ready_engine_response_returns_error_once_then_consumed_error() {
    let mut response = EngineResponse::<GenerateResponse>::ready_err(runtime_command("boom"));

    let first =
        block_on(poll_fn(|cx| Pin::new(&mut response).poll(cx))).expect_err("first ready error");
    let second = block_on(poll_fn(|cx| Pin::new(&mut response).poll(cx)))
        .expect_err("second consumed error");

    assert!(first.to_string().contains("boom"));
    assert!(second.to_string().contains("already consumed"));
}

#[test]
fn token_channel_is_optional_and_streams_until_sender_is_dropped() {
    let (disabled_tx, disabled_rx) = token_channel(false);
    assert!(disabled_tx.is_none());
    assert!(disabled_rx.is_none());

    let (enabled_tx, enabled_rx) = token_channel(true);
    let sender = enabled_tx.expect("enabled sender");
    let mut receiver = enabled_rx.expect("enabled receiver");
    sender
        .unbounded_send(token_batch("a"))
        .expect("send token batch");
    drop(sender);

    assert_eq!(block_on(receiver.next()).expect("token batch").text, "a");
    assert!(block_on(receiver.next()).is_none());
}

#[test]
fn ready_receiver_resolves_preloaded_result() {
    let receiver = ready_receiver::<i32>(Ok(42));

    let value = block_on(receiver)
        .expect("receiver should resolve")
        .expect("result should be ok");

    assert_eq!(value, 42);
}

#[test]
fn closed_engine_query_errors_and_preserves_requested_token_stream() {
    let engine = closed_engine();
    let run = engine.query(QueryRequest::new("hello").emit_tokens(true));
    let (tokens, response) = run.into_parts();

    assert!(tokens.is_some());
    let error = block_on(response).expect_err("closed query");
    assert!(error.to_string().contains("engine thread is closed"));
}

#[test]
fn closed_engine_embed_response_future_errors() {
    let engine = closed_engine();
    let request = EmbedRequest {
        input: "hello".to_string(),
        options: EmbedOptions::default(),
    };

    let error = block_on(engine.embed(request).into_response()).expect_err("closed embed");

    assert!(error.to_string().contains("engine thread is closed"));
}

#[test]
fn closed_engine_state_errors_close_is_idempotent_and_subscribe_registers() {
    let engine = closed_engine();

    let error = block_on(engine.state()).expect_err("closed state");
    assert!(error.to_string().contains("engine thread is closed"));
    block_on(engine.close()).expect("close on closed channel is ok");

    let _events = engine.subscribe_events();
    assert_eq!(
        engine
            .inner
            .event_subscribers
            .lock()
            .expect("subscribers")
            .len(),
        1
    );
}

/// Model-backed live-prefix cache smoke. Ignored by default because it loads
/// the repo-root decoder fixture and runs llama.cpp.
#[test]
#[ignore]
fn decoder_live_prefix_does_not_reuse_repeated_prompt() {
    let fixture = repo_fixture("Qwen3.5-0.8B-Q4_0.gguf");
    assert!(
        fixture.exists(),
        "repo-root Qwen3.5-0.8B-Q4_0.gguf must exist"
    );
    let prompt = "Explain KV cache reuse in one sentence.";

    let engine = block_on(SippEngine::load(
        &fixture,
        cache_smoke_config(KvReuseMode::LiveSlotPrefix),
    ))
    .expect("load decoder fixture");

    let cold = block_on(engine.query(cache_query("cache-smoke", prompt))).expect("cold query");
    assert_eq!(cold.stats.cache_source, CacheSource::None);
    assert_eq!(cold.stats.cache_hits, 0);

    let hot = block_on(engine.query(cache_query("cache-smoke", prompt))).expect("hot query");
    assert_eq!(hot.stats.cache_source, CacheSource::None);
    assert_eq!(hot.stats.cache_hits, 0);
    assert_eq!(hot.stats.prefill_tokens, hot.stats.input_tokens);

    block_on(engine.close()).expect("close engine");
}

/// Model-backed prompt snapshot smoke. Ignored by default because it loads
/// the repo-root decoder fixture and runs llama.cpp.
#[test]
#[ignore]
fn decoder_snapshot_reports_same_prompt_cache_hits() {
    let fixture = repo_fixture("Qwen3.5-0.8B-Q4_0.gguf");
    assert!(
        fixture.exists(),
        "repo-root Qwen3.5-0.8B-Q4_0.gguf must exist"
    );
    let prompt = "Explain KV cache reuse in one sentence.";

    let engine = block_on(SippEngine::load(
        &fixture,
        cache_smoke_config(KvReuseMode::LiveSlotAndSnapshot),
    ))
    .expect("load decoder fixture");

    let cold = block_on(engine.query(cache_query("snapshot-smoke", prompt))).expect("cold query");
    assert_eq!(cold.stats.cache_source, CacheSource::None);
    assert_eq!(cold.stats.cache_hits, 0);

    let hot = block_on(engine.query(cache_query("snapshot-smoke", prompt))).expect("hot query");
    assert_eq!(
        hot.stats.cache_source,
        CacheSource::Snapshot,
        "hot stats: {:?}",
        hot.stats
    );
    assert!(
        hot.stats.cache_hits > 0,
        "same-prompt snapshot should reuse a prompt prefix: {:?}",
        hot.stats
    );
    assert!(
        hot.stats.prefill_tokens < hot.stats.input_tokens,
        "hot stats: {:?}",
        hot.stats
    );

    block_on(engine.close()).expect("close engine");
}

/// Model-backed disabled-cache smoke. Ignored by default because it loads the
/// repo-root decoder fixture and runs llama.cpp.
#[test]
#[ignore]
fn decoder_disabled_cache_reports_full_prefill() {
    let fixture = repo_fixture("Qwen3.5-0.8B-Q4_0.gguf");
    assert!(
        fixture.exists(),
        "repo-root Qwen3.5-0.8B-Q4_0.gguf must exist"
    );

    let engine = block_on(SippEngine::load(
        &fixture,
        cache_smoke_config(KvReuseMode::Disabled),
    ))
    .expect("load decoder fixture");
    let base_prompt = "Explain KV cache reuse in one sentence.";
    let extended_prompt =
        "Explain KV cache reuse in one sentence. Include one concrete browser benefit.";

    let cold =
        block_on(engine.query(cache_query("cache-disabled", base_prompt))).expect("cold query");
    let hot =
        block_on(engine.query(cache_query("cache-disabled", extended_prompt))).expect("hot query");

    assert_eq!(cold.stats.cache_source, CacheSource::None);
    assert_eq!(hot.stats.cache_source, CacheSource::None);
    assert_eq!(hot.stats.cache_hits, 0);
    assert_eq!(hot.stats.prefill_tokens, hot.stats.input_tokens);

    block_on(engine.close()).expect("close engine");
}

/// End-to-end Phase 3 verification: load the repo-root T5 fixture, confirm
/// classification, run an actual encoder-decoder query, and confirm chat()
/// rejects with `UnsupportedOperation`. Ignored by default because it loads a
/// real model and runs llama.cpp under the test harness — run with
/// `cargo test -p sipp -- --ignored t5_encoder_decoder_end_to_end`.
#[test]
#[ignore]
fn t5_encoder_decoder_end_to_end() {
    use crate::engine::protocol::ModelClass;
    use crate::engine::{ChatMessage, ChatRequest, ChatRole};
    use crate::error::Error;

    let fixture = repo_fixture("t5-small-f16.gguf");
    assert!(
        fixture.exists(),
        "repo-root t5-small-f16.gguf must exist for the encoder-decoder gate"
    );

    let engine = block_on(SippEngine::load(&fixture, NativeRuntimeConfig::default()))
        .expect("load t5-small-f16.gguf");

    let state = block_on(engine.state()).expect("engine state");
    let capabilities = state
        .model
        .as_ref()
        .map(|model| &model.capabilities)
        .expect("model state on loaded engine");
    assert_eq!(capabilities.model_class, ModelClass::EncoderDecoder);
    assert!(capabilities.supports_text_generation);
    assert!(!capabilities.supports_embeddings);
    assert!(
        !capabilities.has_chat_template,
        "T5 has no chat template; chat() must be rejected"
    );

    // chat() must reject before touching the runtime.
    let chat_error = block_on(engine.chat(ChatRequest::new(vec![ChatMessage::new(
        ChatRole::User,
        "hello",
    )])))
    .expect_err("chat() must reject T5");
    assert!(
        matches!(
            &chat_error,
            Error::UnsupportedOperation {
                operation: "chat",
                ..
            }
        ),
        "expected UnsupportedOperation; got: {chat_error:?}"
    );

    // query() against T5 should run the encoder pre-pass + decoder loop and
    // return a non-empty text result.
    let result = block_on(engine.query(QueryRequest::new(
        "translate English to German: Hello, world.",
    )))
    .expect("T5 query");
    assert!(
        !result.text.is_empty(),
        "T5 encoder-decoder query produced empty output"
    );
}
