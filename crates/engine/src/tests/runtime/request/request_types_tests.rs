//! Tests the `runtime::request::request_types` module in `cogentlm-engine`.
//!
//! Covers request defaults and reset semantics with deterministic in-memory
//! request values.

use std::time::Instant;

use crate::engine::protocol::EmbedOptions;
use crate::runtime::config::{RequestSampling, SamplingRuntimePatch};

use super::*;

#[test]
fn generate_request_default_is_pending_and_model_free() {
    let request = GenerateRequest::default();

    assert_eq!(request.id, 0);
    assert_eq!(request.context_key, "");
    assert_eq!(request.original_prompt, "");
    assert_eq!(request.grammar, "");
    assert_eq!(request.json_schema, "");
    assert!(request.stop.is_empty());
    assert!(request.sampling.is_none());
    assert_eq!(request.lifecycle, GenerateRequestLifecycle::Pending);
    assert_eq!(request.cache_mode, KvReuseMode::LiveSlotPrefix);
    assert_eq!(request.cache_source, CacheSource::None);
    assert_eq!(request.first_sampled_token_id, NO_SAMPLED_TOKEN_ID);
    assert!(request.prompt_tokens.is_empty());
    assert!(request.multimodal.is_none());
    assert!(request.embed_options.is_none());
    assert_eq!(request.max_output_tokens, 0);
    assert!(request.enqueued_at.is_none());
    assert!(request.admitted_at.is_none());
    assert!(request.first_token_at.is_none());
    assert!(request.last_token_at.is_none());
    assert!(request.completed_at.is_none());
    assert_eq!(request.emitted_token_count, 0);
    assert_eq!(request.itl_sum_ms, 0.0);
    assert_eq!(request.itl_p99_ms, 0.0);
    assert_eq!(request.e2e_ms, 0.0);
    assert_eq!(request.prefill_ms, 0.0);
    assert_eq!(request.decode_ms, 0.0);
    assert_eq!(request.native_sync_ms, 0.0);
    assert_eq!(request.native_gpu_ms, 0.0);
    assert_eq!(request.native_logic_ms, 0.0);
    assert_eq!(request.input_tokens, 0);
    assert_eq!(request.output_tokens, 0);
    assert_eq!(request.cache_hits, 0);
    assert_eq!(request.prefill_tokens, 0);
    assert!(!request.is_multimodal_turn);
    assert!(!request.emit_tokens);
    assert!(!request.cancel_requested);
}

#[test]
fn generate_request_new_sets_id_context_and_enqueue_time() {
    let request = GenerateRequest::new(7, "ctx");

    assert_eq!(request.id, 7);
    assert_eq!(request.context_key, "ctx");
    assert!(request.enqueued_at.is_some());
    assert_eq!(request.lifecycle, GenerateRequestLifecycle::Pending);
}

#[test]
fn reset_for_queue_preserves_identity_and_clears_runtime_state() {
    let mut request = GenerateRequest::new(7, "ctx");
    request.lifecycle = GenerateRequestLifecycle::Failed;
    request.admitted_at = Some(Instant::now());
    request.first_token_at = Some(Instant::now());
    request.last_token_at = Some(Instant::now());
    request.completed_at = Some(Instant::now());
    request.emitted_token_count = 5;
    request.itl_sum_ms = 1.0;
    request.itl_p99_ms = 2.0;
    request.e2e_ms = 3.0;
    request.prefill_ms = 4.0;
    request.decode_ms = 5.0;
    request.native_sync_ms = 6.0;
    request.native_gpu_ms = 7.0;
    request.native_logic_ms = 8.0;
    request.input_tokens = 3;
    request.output_tokens = 2;
    request.cache_mode = KvReuseMode::Disabled;
    request.cache_source = CacheSource::Live;
    request.cache_hits = 2;
    request.prefill_tokens = 3;
    request.first_sampled_token_id = 99;
    request.cancel_requested = true;

    request.reset_for_queue();

    assert_eq!(request.id, 7);
    assert_eq!(request.context_key, "ctx");
    assert_eq!(request.lifecycle, GenerateRequestLifecycle::Pending);
    assert!(request.admitted_at.is_none());
    assert!(request.first_token_at.is_none());
    assert!(request.last_token_at.is_none());
    assert!(request.completed_at.is_none());
    assert_eq!(request.emitted_token_count, 0);
    assert_eq!(request.itl_sum_ms, 0.0);
    assert_eq!(request.itl_p99_ms, 0.0);
    assert_eq!(request.e2e_ms, 0.0);
    assert_eq!(request.prefill_ms, 0.0);
    assert_eq!(request.decode_ms, 0.0);
    assert_eq!(request.native_sync_ms, 0.0);
    assert_eq!(request.native_gpu_ms, 0.0);
    assert_eq!(request.native_logic_ms, 0.0);
    assert_eq!(request.input_tokens, 0);
    assert_eq!(request.output_tokens, 0);
    assert_eq!(request.cache_mode, KvReuseMode::Disabled);
    assert_eq!(request.cache_source, CacheSource::None);
    assert_eq!(request.cache_hits, 0);
    assert_eq!(request.prefill_tokens, 0);
    assert_eq!(request.first_sampled_token_id, NO_SAMPLED_TOKEN_ID);
    assert!(!request.cancel_requested);
}

#[test]
fn multimodal_payload_and_request_overrides_are_plain_value_fields() {
    let payload = MultimodalPayload {
        image_buffers: vec![vec![1, 2, 3]],
    };
    let mut request = GenerateRequest::new(9, "embed");
    request.original_prompt = "prompt".to_string();
    request.grammar = "root ::= \"x\"".to_string();
    request.json_schema = "{}".to_string();
    request.stop = vec!["</s>".to_string()];
    request.sampling = Some(RequestSampling::Patch(SamplingRuntimePatch {
        temperature: Some(0.2),
        top_p: None,
    }));
    request.prompt_tokens = vec![1, 2, 3];
    request.multimodal = Some(payload.clone());
    request.embed_options = Some(EmbedOptions {
        normalize: false,
        context_key: Some("embedding".to_string()),
    });
    request.max_output_tokens = 4;
    request.emit_tokens = true;
    request.is_multimodal_turn = true;

    assert_eq!(request.original_prompt, "prompt");
    assert_eq!(request.grammar, "root ::= \"x\"");
    assert_eq!(request.json_schema, "{}");
    assert_eq!(request.stop, vec!["</s>"]);
    assert!(matches!(
        request.sampling,
        Some(RequestSampling::Patch(SamplingRuntimePatch {
            temperature: Some(0.2),
            top_p: None
        }))
    ));
    assert_eq!(request.prompt_tokens, vec![1, 2, 3]);
    assert_eq!(request.multimodal, Some(payload));
    assert_eq!(
        request.embed_options,
        Some(EmbedOptions {
            normalize: false,
            context_key: Some("embedding".to_string())
        })
    );
    assert_eq!(request.max_output_tokens, 4);
    assert!(request.emit_tokens);
    assert!(request.is_multimodal_turn);
}
