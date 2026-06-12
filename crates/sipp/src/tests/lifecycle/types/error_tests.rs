//! Tests the `lifecycle::types::error` module in `sipp`.
//!
//! Covers model-error display text and conversions from crate and GGUF errors
//! with deterministic value fixtures.

use super::*;

#[test]
fn model_error_display_messages_are_stable() {
    let cases = [
        (
            ModelError::InvalidModelSource("empty".to_string()),
            "invalid model source: empty",
        ),
        (
            ModelError::RemoteUnavailable("https://example.test/model.gguf".to_string()),
            "remote model loading is not available in this runtime: https://example.test/model.gguf",
        ),
        (
            ModelError::UnsupportedOperation {
                operation: "embed",
                reason: "not loaded".to_string(),
            },
            "unsupported operation embed: not loaded",
        ),
    ];

    for (error, expected) in cases {
        assert_eq!(error.to_string(), expected);
    }
}

#[test]
fn crate_error_conversion_preserves_unsupported_operation() {
    let error = ModelError::from(crate::error::Error::UnsupportedOperation {
        operation: "chat",
        reason: "missing template".to_string(),
    });

    assert!(matches!(
        error,
        ModelError::UnsupportedOperation {
            operation: "chat",
            reason
        } if reason == "missing template"
    ));
}

#[test]
fn gguf_error_conversion_maps_each_variant_to_model_error() {
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
    assert!(matches!(
        ModelError::from(crate::shard::GgufError::Io(io_error)),
        ModelError::Io(error) if error.kind() == std::io::ErrorKind::NotFound
    ));

    assert!(matches!(
        ModelError::from(crate::shard::GgufError::Invalid("bad metadata".to_string())),
        ModelError::InvalidGgufMetadata(message) if message == "bad metadata"
    ));
    assert!(matches!(
        ModelError::from(crate::shard::GgufError::UnsupportedVersion(1)),
        ModelError::UnsupportedGgufVersion(1)
    ));
    assert!(matches!(
        ModelError::from(crate::shard::GgufError::MetadataTooLarge { max_bytes: 4096 }),
        ModelError::GgufMetadataTooLarge { max_bytes: 4096 }
    ));
    assert!(matches!(
        ModelError::from(crate::shard::GgufError::AlreadySplit(3)),
        ModelError::InvalidGgufMetadata(message)
            if message == "source GGUF is already split into 3 files"
    ));
}

#[test]
fn crate_error_conversion_wraps_other_runtime_errors() {
    let error = ModelError::from(crate::error::Error::RuntimeCommand(
        "native failed".to_string(),
    ));

    assert!(matches!(
        error,
        ModelError::Runtime(message) if message == "runtime command failed: native failed"
    ));
}
