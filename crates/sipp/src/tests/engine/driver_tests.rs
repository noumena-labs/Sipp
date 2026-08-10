//! Tests the `engine::driver` module in `sipp`.
//!
//! Covers driver futures, command handling, event emission, and request mapping with model-free channels.

use super::*;
use crate::core::{TokenBatch, TokenEmissionStats};
use crate::engine::{
    EmbedOptions, GenerateOptions, SamplingRuntimeOverride, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};
use crate::runtime::request::GenerateResponse;
use futures::executor::block_on;
use futures::future::poll_fn;
use futures::StreamExt;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

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
fn closed_engine_listen_uses_the_generation_error_and_has_no_token_stream() {
    let engine = closed_engine();
    let run = engine.listen(ListenRequest {
        audio: vec![1, 2, 3],
        language: Some("en".to_string()),
        max_tokens: 32,
    });
    let (tokens, response) = run.into_parts();

    assert!(tokens.is_none());
    let error = block_on(response).expect_err("closed listen");
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
