//! Unit tests for the parent module.

use super::super::*;

enum TestValue<'a> {
    String(&'a str),
    Bool(bool),
}

fn gguf(entries: &[(&str, TestValue<'_>)]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(24);
    push_u32(&mut bytes, GGUF_MAGIC);
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 0);
    push_u64(
        &mut bytes,
        test_u64_from_usize(entries.len(), "entry count"),
    );
    for (key, value) in entries {
        push_string(&mut bytes, key);
        match value {
            TestValue::String(value) => {
                push_u32(&mut bytes, GgufValueType::String.as_u32());
                push_string(&mut bytes, value);
            }
            TestValue::Bool(value) => {
                push_u32(&mut bytes, GgufValueType::Bool.as_u32());
                bytes.push(u8::from(*value));
            }
        }
    }
    bytes
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, test_u64_from_usize(value.len(), "string length"));
    bytes.extend_from_slice(value.as_bytes());
}

fn test_u64_from_usize(value: usize, name: &str) -> u64 {
    u64::try_from(value).expect(name)
}

#[test]
fn detects_lfm_vision_base_model() {
    let detection = detect_model_from_gguf_bytes(
        "base.gguf",
        &gguf(&[
            ("general.architecture", TestValue::String("lfm2")),
            ("clip.has_vision_encoder", TestValue::Bool(true)),
        ]),
    )
    .expect("detection");

    assert_eq!(
        detection.detection_method,
        ModelDetectionMethod::GgufMetadata
    );
    assert_eq!(detection.inspection.role, AssetRole::Model);
    assert!(detection.inspection.vision_capable);
    assert_eq!(
        detection.inspection.compatible_vision_projector_types,
        vec!["lfm2"]
    );
}

#[test]
fn detects_minicpm_vision_base_model() {
    let detection = detect_model_from_gguf_bytes(
        "minicpm.gguf",
        &gguf(&[
            ("general.architecture", TestValue::String("minicpm")),
            ("clip.has_vision_encoder", TestValue::Bool(true)),
        ]),
    )
    .expect("detection");

    assert_eq!(detection.inspection.role, AssetRole::Model);
    assert!(detection.inspection.vision_capable);
    assert_eq!(
        detection.inspection.compatible_vision_projector_types,
        vec!["resampler", "minicpmv4_6"]
    );
}

#[test]
fn detects_projector_from_mmproj_metadata() {
    let detection = detect_model_from_gguf_bytes(
        "mmproj.gguf",
        &gguf(&[
            ("general.type", TestValue::String("mmproj")),
            ("general.architecture", TestValue::String("clip")),
            ("clip.projector_type", TestValue::String("lfm2")),
            ("clip.has_vision_encoder", TestValue::Bool(true)),
        ]),
    )
    .expect("detection");

    assert_eq!(detection.inspection.role, AssetRole::Projector);
    assert_eq!(
        detection.inspection.provided_vision_projector_type,
        Some("lfm2".to_string())
    );
}

#[test]
fn non_gguf_bytes_are_unknown() {
    let detection = detect_model_from_gguf_bytes("bad.bin", b"not a gguf").expect("detection");

    assert_eq!(detection.detection_method, ModelDetectionMethod::None);
    assert_eq!(detection.inspection, AssetInspection::unknown());
}

#[test]
fn truncated_gguf_metadata_is_typed_error() {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, GGUF_MAGIC);
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 0);
    push_u64(&mut bytes, 1);
    push_u64(&mut bytes, 12);
    bytes.extend_from_slice(b"general.type");

    let error = inspect_gguf_metadata(&bytes).expect_err("truncated value type");

    assert!(
        matches!(error, ModelError::InvalidGgufMetadata(message) if message.contains("truncated"))
    );
}

#[test]
fn oversized_gguf_string_length_is_typed_error() {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, GGUF_MAGIC);
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 0);
    push_u64(&mut bytes, 1);
    push_u64(&mut bytes, u64::MAX);

    let error = inspect_gguf_metadata(&bytes).expect_err("oversized key");

    assert!(matches!(&error, ModelError::InvalidGgufMetadata(message)
            if message.contains("length does not fit usize")
                || message.contains("truncated")
                || message.contains("offset overflow")));
}
