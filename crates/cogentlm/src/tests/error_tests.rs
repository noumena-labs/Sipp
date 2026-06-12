//! Tests the crate-level `error` module in `cogentlm`.
//!
//! Covers typed error display text and source conversion behavior with
//! deterministic value-only fixtures.

use std::ffi::CString;

use super::*;

#[test]
fn error_display_messages_are_actionable() {
    let cases = [
        (
            Error::ModelLoad {
                path: "model.gguf".to_string(),
            },
            "failed to load model from model.gguf",
        ),
        (
            Error::BatchCapacity {
                capacity: 4,
                requested: 5,
            },
            "batch capacity exceeded: capacity=4, requested=5",
        ),
        (
            Error::PromptTooLong {
                prompt_tokens: 9,
                context_tokens: 8,
            },
            "prompt has 9 tokens but context allows 8",
        ),
        (
            Error::UnsupportedOperation {
                operation: "embed",
                reason: "decoder-only".to_string(),
            },
            "unsupported operation embed: decoder-only",
        ),
    ];

    for (error, expected) in cases {
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn interior_nul_converts_from_cstring_error() {
    let error = CString::new(b"a\0b".to_vec()).expect_err("interior nul");
    let error = Error::from(error);

    assert!(matches!(error, Error::InteriorNul(_)));
    assert_eq!(error.to_string(), "string contains an interior NUL byte");
}
