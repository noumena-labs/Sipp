//! Tests the `runtime::inference_runtime::request::api` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::inference_runtime::runtime_tests::test_runtime;
use crate::runtime::request::GenerateRequest;

use super::*;

#[test]
fn normalize_context_key_uses_default_for_empty_values() {
    assert_eq!(normalize_context_key(""), DEFAULT_PROMPT_CONTEXT_KEY);
    assert_eq!(normalize_context_key("ctx"), "ctx");
}

#[test]
fn generate_request_maps_all_fields_and_normalizes_stops() {
    let request = generate_request(GenerateRequestFields {
        request_id: 3,
        context_key: "ctx".to_string(),
        prompt: "prompt".to_string(),
        prompt_tokens: vec![1, 2],
        n_tokens_predict: 5,
        grammar: "root ::= \"ok\"".to_string(),
        json_schema: "{}".to_string(),
        stop: vec!["zz".to_string(), "aa".to_string(), String::new()],
        sampling: None,
        emit_tokens: true,
    });

    assert_eq!(request.id, 3);
    assert_eq!(request.context_key, "ctx");
    assert_eq!(request.original_prompt, "prompt");
    assert_eq!(request.prompt_tokens, vec![1, 2]);
    assert_eq!(request.max_output_tokens, 5);
    assert_eq!(request.grammar, "root ::= \"ok\"");
    assert_eq!(request.json_schema, "{}");
    assert_eq!(request.stop, vec!["aa", "zz"]);
    assert!(request.emit_tokens);
}

#[test]
fn request_tokenization_flags_match_text_and_multimodal_modes() {
    assert_eq!(
        request_tokenization_flags_for_tests("text"),
        Some((true, true))
    );
    assert_eq!(
        request_tokenization_flags_for_tests("multimodal"),
        Some((false, false))
    );
    assert_eq!(request_tokenization_flags_for_tests("bad"), None);
}

#[test]
fn normalize_stop_sequences_drops_empty_values_and_deduplicates() {
    assert_eq!(
        normalize_stop_sequences(vec![
            "z".to_string(),
            String::new(),
            "a".to_string(),
            "z".to_string(),
        ]),
        vec!["a", "z"]
    );
}

#[test]
fn next_generate_request_id_rejects_overflow() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.next_request_id = GenerateRequestId::MAX;

    let error = runtime
        .next_generate_request_id()
        .expect_err("request id overflow");

    assert!(matches!(
        error,
        Error::InvalidRequest(message) if message.contains("request id overflow")
    ));
}

#[test]
fn enqueue_prepared_request_rejects_zero_request_id() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    let mut request = GenerateRequest::new(0, "ctx");
    request.prompt_tokens = vec![1, 2, 3];

    let error = runtime
        .enqueue_prepared_request(request)
        .expect_err("zero request id is invalid");

    assert!(matches!(
        error,
        Error::InvalidRequest(message) if message.contains("failed to enqueue request")
    ));
    assert_eq!(runtime.total_input_tokens, 3);
}

#[test]
fn enqueue_prepared_request_saturates_total_input_token_counter() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());
    runtime.total_input_tokens = usize::MAX - 1;
    let mut request = GenerateRequest::new(8, "ctx");
    request.prompt_tokens = vec![1, 2, 3];

    let request_id = runtime
        .enqueue_prepared_request(request)
        .expect("enqueue request");

    assert_eq!(request_id, 8);
    assert_eq!(runtime.total_input_tokens, usize::MAX);
}

#[test]
fn enqueue_generation_requests_reject_empty_runtime_before_tokenization() {
    let mut runtime = test_runtime(NativeRuntimeConfig::default());

    assert!(matches!(
        runtime.enqueue_request("ctx", "prompt", 1, "", "", Vec::new(), None, false),
        Err(Error::RuntimeNotReady)
    ));
    assert!(matches!(
        runtime.enqueue_multimodal_request(
            "ctx",
            "prompt",
            1,
            vec![vec![1, 2, 3]],
            "",
            "",
            Vec::new(),
            None,
            false
        ),
        Err(Error::RuntimeNotReady)
    ));
}
