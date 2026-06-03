//! Tests the `runtime::request::response_types` module in `cogentlm-engine`.
//!
//! Covers response status strings, default payloads, and terminal constructor
//! behavior with deterministic in-memory values.

use super::*;

#[test]
fn generate_response_status_strings_cover_all_variants() {
    let cases = [
        (GenerateResponseStatus::Pending, "pending"),
        (GenerateResponseStatus::Completed, "completed"),
        (GenerateResponseStatus::Cancelled, "cancelled"),
        (GenerateResponseStatus::Failed, "failed"),
    ];

    for (status, expected) in cases {
        assert_eq!(status.as_str(), expected);
    }
}

#[test]
fn response_output_default_is_empty_text() {
    assert_eq!(
        ResponseOutput::default(),
        ResponseOutput::Text(String::new())
    );
}

#[test]
fn response_output_embedding_preserves_values_pooling_and_normalization_flag() {
    let output = ResponseOutput::Embedding {
        values: vec![0.25, -0.5, 1.0],
        pooling: PoolingType::Mean,
        normalized: true,
    };

    assert_eq!(
        output,
        ResponseOutput::Embedding {
            values: vec![0.25, -0.5, 1.0],
            pooling: PoolingType::Mean,
            normalized: true
        }
    );
}

#[test]
fn generate_response_default_is_pending_empty_text_without_error() {
    let response = GenerateResponse::default();

    assert_eq!(response.request_id, 0);
    assert_eq!(response.status, GenerateResponseStatus::Pending);
    assert_eq!(response.output, ResponseOutput::default());
    assert_eq!(response.error_message, "");
    assert_eq!(response.runtime_observability.input_tokens, 0);
    assert_eq!(response.runtime_observability.output_tokens, 0);
}

#[test]
fn terminal_response_preserves_status_output_and_error() {
    let response = GenerateResponse::terminal(
        7,
        GenerateResponseStatus::Completed,
        ResponseOutput::Text("done".to_string()),
        "ignored",
    );

    assert_eq!(response.request_id, 7);
    assert_eq!(response.status, GenerateResponseStatus::Completed);
    assert_eq!(response.output, ResponseOutput::Text("done".to_string()));
    assert_eq!(response.error_message, "ignored");
}

#[test]
fn cancelled_and_failed_responses_use_empty_text_output() {
    let cancelled = GenerateResponse::cancelled(1, "cancelled");
    let failed = GenerateResponse::failed(2, "failed");

    assert_eq!(cancelled.status, GenerateResponseStatus::Cancelled);
    assert_eq!(cancelled.output, ResponseOutput::default());
    assert_eq!(cancelled.error_message, "cancelled");
    assert_eq!(failed.status, GenerateResponseStatus::Failed);
    assert_eq!(failed.output, ResponseOutput::default());
    assert_eq!(failed.error_message, "failed");
}
