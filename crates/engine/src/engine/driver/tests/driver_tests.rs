//! Unit tests for the parent module.

use super::super::*;
use crate::engine::{
    CacheSource, GenerateOptions, GpuLayerConfig, KvReuseMode, NativeRuntimeConfig,
    RequestSampling, SamplingRuntimeConfig, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};
use futures::executor::block_on;
use std::path::PathBuf;

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
    let Some(RequestSampling::Full(sampling)) = &options.sampling else {
        panic!("generate options should map to a full sampling override");
    };
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
    assert_send::<CogentEngine>();
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

    let engine = block_on(CogentEngine::load(
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

    let engine = block_on(CogentEngine::load(
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

    let engine = block_on(CogentEngine::load(
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
/// `cargo test -p cogentlm-engine -- --ignored t5_encoder_decoder_end_to_end`.
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

    let engine = block_on(CogentEngine::load(&fixture, NativeRuntimeConfig::default()))
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
