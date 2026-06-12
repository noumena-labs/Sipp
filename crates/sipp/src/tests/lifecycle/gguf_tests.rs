//! Tests the `lifecycle::gguf` module in `sipp`.
//!
//! Covers the GGUF detection wrapper's deterministic unknown-detection path on
//! arbitrary in-memory bytes without reading local model fixtures.

use super::*;

#[test]
fn detect_model_from_arbitrary_bytes_returns_unknown_detection() {
    let detection =
        detect_model_from_gguf_bytes("bad.gguf", b"not a gguf").expect("unknown detection");

    assert_eq!(detection.model_name, "bad.gguf");
    assert_eq!(
        detection.inspection.role,
        crate::lifecycle::AssetRole::Unknown
    );
    assert_eq!(detection.inspection.architecture, None);
    assert_eq!(
        detection.detection_method,
        crate::lifecycle::ModelDetectionMethod::None
    );
}
