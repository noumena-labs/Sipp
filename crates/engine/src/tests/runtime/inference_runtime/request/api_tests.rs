//! Tests the `runtime::inference_runtime::request::api` module in `cogentlm-engine`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use crate::runtime::inference_runtime::request::api::{
    normalize_stop_sequences, request_tokenization_flags_for_tests,
};

#[test]
fn normalize_stop_sequences_drops_empty_and_deduplicates() {
    let normalized = normalize_stop_sequences(vec![
        "zz".to_string(),
        String::new(),
        "aa".to_string(),
        "zz".to_string(),
    ]);

    assert_eq!(normalized, ["aa", "zz"]);
}

#[test]
fn request_tokenization_modes_preserve_text_and_multimodal_flags() {
    assert_eq!(
        request_tokenization_flags_for_tests("text"),
        Some((true, true))
    );
    assert_eq!(
        request_tokenization_flags_for_tests("multimodal"),
        Some((false, false))
    );
    assert_eq!(request_tokenization_flags_for_tests("unknown"), None);
}
