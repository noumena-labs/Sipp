//! Tests the `inspection` module in `sipp::shard`.
//!
//! Covers deterministic GGUF metadata inspection, model/projector detection,
//! prefix truncation boundaries, value skipping, and vision architecture
//! compatibility without native/model execution.

use super::*;
use crate::shard::support::{
    metadata_gguf, metadata_gguf_version, unique_temp_dir, MetadataValue as FixtureValue,
};

use std::fs;

#[test]
fn path_inspection_reads_file_prefix_and_reports_metadata() {
    let root = unique_temp_dir();
    fs::create_dir_all(&root).expect("temp dir");
    let path = root.join("model.gguf");
    fs::write(
        &path,
        metadata_gguf(&[("general.architecture", FixtureValue::String("LLAMA"))]),
    )
    .expect("write gguf");

    let metadata = inspect_gguf_metadata_path(&path)
        .expect("path inspection")
        .expect("metadata");

    assert_eq!(metadata.general_architecture.as_deref(), Some("llama"));
    fs::remove_dir_all(root).ok();
}

#[test]
fn short_and_non_gguf_bytes_are_ignored() {
    assert_eq!(inspect_gguf_metadata(b"short").expect("short"), None);
    assert_eq!(
        inspect_gguf_metadata(b"not a gguf header with enough bytes").expect("non gguf"),
        None
    );
}

#[test]
fn unsupported_version_is_typed_error() {
    let error =
        inspect_gguf_metadata(&metadata_gguf_version(99, &[])).expect_err("unsupported version");

    assert!(matches!(error, GgufError::UnsupportedVersion(99)));
}

#[test]
fn inspects_target_keys_and_normalizes_optional_strings() {
    let metadata = inspect_gguf_metadata(&metadata_gguf(&[
        ("general.type", FixtureValue::String("  MODEL  ")),
        ("general.architecture", FixtureValue::String(" Qwen2VL ")),
        ("general.pooling_type", FixtureValue::Uint32(2)),
        ("clip.projector_type", FixtureValue::String(" Resampler ")),
        (
            "clip.vision.projector_type",
            FixtureValue::String(" Qwen2VL_Merger "),
        ),
        ("clip.has_vision_encoder", FixtureValue::Bool(true)),
    ]))
    .expect("inspection")
    .expect("metadata");

    assert_eq!(metadata.general_type.as_deref(), Some("model"));
    assert_eq!(metadata.general_architecture.as_deref(), Some("qwen2vl"));
    assert_eq!(metadata.pooling_type, Some(2));
    assert_eq!(metadata.clip_projector_type.as_deref(), Some("resampler"));
    assert_eq!(
        metadata.clip_vision_projector_type.as_deref(),
        Some("qwen2vl_merger")
    );
    assert_eq!(metadata.clip_has_vision_encoder, Some(true));
}

#[test]
fn blank_target_strings_normalize_to_none() {
    let metadata = inspect_gguf_metadata(&metadata_gguf(&[
        ("general.type", FixtureValue::String("   ")),
        ("general.architecture", FixtureValue::String("\t")),
    ]))
    .expect("inspection")
    .expect("metadata");

    assert_eq!(metadata.general_type, None);
    assert_eq!(metadata.general_architecture, None);
}

#[test]
fn scans_suffix_pooling_keys() {
    let metadata = inspect_gguf_metadata(&metadata_gguf(&[
        ("general.architecture", FixtureValue::String("bert")),
        ("bert.pooling_type", FixtureValue::Uint32(1)),
    ]))
    .expect("inspection")
    .expect("metadata");

    assert_eq!(metadata.general_architecture.as_deref(), Some("bert"));
    assert_eq!(metadata.pooling_type, Some(1));
}

#[test]
fn stops_early_after_useful_metadata_before_large_tokenizer_payload() {
    let metadata = inspect_gguf_metadata(&metadata_gguf(&[
        ("general.architecture", FixtureValue::String("llama")),
        (
            "tokenizer.ggml.tokens",
            FixtureValue::ArrayString(&["token-a", "token-b"]),
        ),
        ("general.type", FixtureValue::String("model")),
    ]))
    .expect("inspection")
    .expect("metadata");

    assert_eq!(metadata.general_architecture.as_deref(), Some("llama"));
    assert_eq!(metadata.general_type, None);
    assert_eq!(metadata.scanned_key_count, 2);
    assert_eq!(
        metadata.stopped_early_at_key.as_deref(),
        Some("tokenizer.ggml.tokens")
    );
}

#[test]
fn early_stop_keys_are_skipped_when_no_useful_metadata_exists() {
    let metadata = inspect_gguf_metadata(&metadata_gguf(&[
        ("tokenizer.ggml.scores", FixtureValue::ArrayU32(&[1, 2, 3])),
        ("general.type", FixtureValue::String("model")),
    ]))
    .expect("inspection")
    .expect("metadata");

    assert_eq!(metadata.general_type.as_deref(), Some("model"));
    assert_eq!(metadata.scanned_key_count, 2);
    assert_eq!(metadata.stopped_early_at_key, None);
}

#[test]
fn skips_non_target_scalar_string_and_array_values() {
    let metadata = inspect_gguf_metadata(&metadata_gguf(&[
        ("ignored.u8", FixtureValue::Uint8(7)),
        ("ignored.u16", FixtureValue::Uint16(8)),
        ("ignored.i32", FixtureValue::Int32(-1)),
        ("ignored.u64", FixtureValue::Uint64(9)),
        ("ignored.string", FixtureValue::String("skip me")),
        ("ignored.array", FixtureValue::ArrayString(&["a", "b"])),
        ("general.architecture", FixtureValue::String("llama")),
    ]))
    .expect("inspection")
    .expect("metadata");

    assert_eq!(metadata.general_architecture.as_deref(), Some("llama"));
    assert_eq!(metadata.scanned_key_count, 7);
}

#[test]
fn target_keys_with_mismatched_types_are_skipped() {
    let metadata = inspect_gguf_metadata(&metadata_gguf(&[
        ("general.type", FixtureValue::Uint64(42)),
        ("general.architecture", FixtureValue::ArrayU32(&[1, 2])),
        ("clip.has_vision_encoder", FixtureValue::String("true")),
        ("general.pooling_type", FixtureValue::String("1")),
    ]))
    .expect("inspection")
    .expect("metadata");

    assert_eq!(metadata.general_type, None);
    assert_eq!(metadata.general_architecture, None);
    assert_eq!(metadata.clip_has_vision_encoder, None);
    assert_eq!(metadata.pooling_type, None);
}

#[test]
fn invalid_utf8_target_value_and_non_io_mapping_are_typed_errors() {
    let mut bytes = metadata_header(1);
    push_string(&mut bytes, "general.type");
    push_u32(&mut bytes, GgufValueType::String as u32);
    push_u64(&mut bytes, 1);
    bytes.push(0xff);

    let error = inspect_gguf_metadata(&bytes).expect_err("invalid target value");
    assert!(matches!(
        error,
        GgufError::Invalid(message) if message == "string is not UTF-8"
    ));

    let mapped = map_metadata_error(GgufError::Invalid("kept".to_string()), 1);
    assert!(matches!(
        mapped,
        GgufError::Invalid(message) if message == "kept"
    ));
}

#[test]
fn invalid_value_type_is_typed_error() {
    let mut bytes = metadata_header(1);
    push_string(&mut bytes, "general.type");
    push_u32(&mut bytes, 99);

    let error = inspect_gguf_metadata(&bytes).expect_err("invalid value type");

    assert!(matches!(
        error,
        GgufError::Invalid(message) if message == "unknown value type 99"
    ));
}

#[test]
fn nested_and_overflowing_arrays_are_typed_errors() {
    let nested = inspect_gguf_metadata(&metadata_gguf(&[(
        "ignored.array",
        FixtureValue::ArrayHeader {
            item_type: GgufValueType::Array,
            len: 0,
        },
    )]))
    .expect_err("nested array");

    assert!(matches!(
        nested,
        GgufError::Invalid(message) if message == "nested GGUF arrays are not supported"
    ));

    let overflowing_len = (usize::MAX / 8) as u64 + 1;
    let overflow = inspect_gguf_metadata(&metadata_gguf(&[(
        "ignored.array",
        FixtureValue::ArrayHeader {
            item_type: GgufValueType::Uint64,
            len: overflowing_len,
        },
    )]))
    .expect_err("array overflow");

    assert!(matches!(
        overflow,
        GgufError::Invalid(message) if message == "array length overflow"
    ));
}

#[test]
fn truncated_metadata_and_too_large_prefix_are_typed_errors() {
    let mut truncated = metadata_header(1);
    push_string(&mut truncated, "general.type");

    let error = inspect_gguf_metadata(&truncated).expect_err("truncated value type");
    assert!(matches!(error, GgufError::Invalid(message) if message.contains("truncated")));

    let mut too_large = metadata_header(1);
    push_u64(&mut too_large, DEFAULT_MAX_PREFIX_BYTES as u64);
    too_large.resize(DEFAULT_MAX_PREFIX_BYTES, 0);

    let error = inspect_gguf_metadata(&too_large).expect_err("large prefix");
    assert!(matches!(
        error,
        GgufError::MetadataTooLarge { max_bytes } if max_bytes == DEFAULT_MAX_PREFIX_BYTES
    ));
}

#[test]
fn invalid_utf8_key_and_oversized_key_lengths_are_typed_errors() {
    let mut invalid_utf8 = metadata_header(1);
    push_u64(&mut invalid_utf8, 1);
    invalid_utf8.push(0xff);

    let error = inspect_gguf_metadata(&invalid_utf8).expect_err("utf8 key");
    assert!(matches!(
        error,
        GgufError::Invalid(message) if message == "string is not UTF-8"
    ));

    let mut oversized = metadata_header(1);
    push_u64(&mut oversized, u64::MAX);

    let error = inspect_gguf_metadata(&oversized).expect_err("oversized key");
    assert!(matches!(&error, GgufError::Invalid(message)
            if message.contains("length does not fit usize")
                || message.contains("truncated")
                || message.contains("offset overflow")));
}

#[test]
fn non_gguf_bytes_are_unknown_detection_with_original_name() {
    let detection = detect_model_from_gguf_bytes("bad.bin", b"not a gguf").expect("detection");

    assert_eq!(detection.detection_method, ModelDetectionMethod::None);
    assert_eq!(detection.inspection, AssetInspection::unknown());
    assert_eq!(detection.model_name, "bad.bin");
}

#[test]
fn valid_gguf_without_role_metadata_is_unknown_and_normalizes_blank_name() {
    let detection = detect_model_from_gguf_bytes("   ", &metadata_gguf(&[])).expect("detection");

    assert_eq!(detection.detection_method, ModelDetectionMethod::None);
    assert_eq!(detection.inspection.role, AssetRole::Unknown);
    assert_eq!(detection.model_name, "model.gguf");
    assert_eq!(detection.model_type, None);
    assert_eq!(detection.model_architecture, None);
}

#[test]
fn model_type_without_architecture_detects_non_vision_model() {
    let detection = detect_model_from_gguf_bytes(
        "base.gguf",
        &metadata_gguf(&[("general.type", FixtureValue::String("model"))]),
    )
    .expect("detection");

    assert_eq!(
        detection.detection_method,
        ModelDetectionMethod::GgufMetadata
    );
    assert_eq!(detection.inspection.role, AssetRole::Model);
    assert!(!detection.inspection.vision_capable);
    assert!(detection
        .inspection
        .compatible_vision_projector_types
        .is_empty());
}

#[test]
fn clip_encoder_flag_alone_detects_vision_model() {
    let detection = detect_model_from_gguf_bytes(
        "vision.gguf",
        &metadata_gguf(&[("clip.has_vision_encoder", FixtureValue::Bool(true))]),
    )
    .expect("detection");

    assert_eq!(detection.inspection.role, AssetRole::Model);
    assert!(detection.inspection.vision_capable);
}

#[test]
fn projector_detection_uses_mmproj_clip_architecture_and_projector_type() {
    let mmproj = detect_model_from_gguf_bytes(
        "mmproj.gguf",
        &metadata_gguf(&[
            ("general.type", FixtureValue::String("mmproj")),
            ("general.architecture", FixtureValue::String("llama")),
        ]),
    )
    .expect("mmproj");
    assert_eq!(mmproj.inspection.role, AssetRole::Projector);

    let clip = detect_model_from_gguf_bytes(
        "clip.gguf",
        &metadata_gguf(&[("general.architecture", FixtureValue::String("clip"))]),
    )
    .expect("clip");
    assert_eq!(clip.inspection.role, AssetRole::Projector);

    let projector_type = detect_model_from_gguf_bytes(
        "projector.gguf",
        &metadata_gguf(&[("clip.projector_type", FixtureValue::String("lfm2"))]),
    )
    .expect("projector type");
    assert_eq!(projector_type.inspection.role, AssetRole::Projector);
    assert_eq!(
        projector_type.inspection.provided_vision_projector_type,
        Some("lfm2".to_string())
    );
}

#[test]
fn vision_projector_type_takes_precedence_over_legacy_projector_type() {
    let detection = detect_model_from_gguf_bytes(
        "projector.gguf",
        &metadata_gguf(&[
            ("clip.projector_type", FixtureValue::String("legacy")),
            ("clip.vision.projector_type", FixtureValue::String("modern")),
        ]),
    )
    .expect("detection");

    assert_eq!(detection.inspection.role, AssetRole::Projector);
    assert_eq!(
        detection.inspection.provided_vision_projector_type,
        Some("modern".to_string())
    );
}

#[test]
fn detects_every_known_vision_architecture_mapping() {
    for (architecture, has_encoder, expected) in [
        ("cogvlm", false, vec!["cogvlm"]),
        ("gemma3", true, vec!["gemma3"]),
        ("gemma3n", true, vec!["gemma3nv"]),
        ("gemma4", true, vec!["gemma4v"]),
        ("hunyuan_vl", false, vec!["hunyuanvl"]),
        ("lfm2", true, vec!["lfm2"]),
        ("llama4", true, vec!["llama4"]),
        ("minicpm", true, vec!["resampler", "minicpmv4_6"]),
        ("minicpm3", true, vec!["resampler", "minicpmv4_6"]),
        ("paddleocr", false, vec!["paddleocr"]),
        ("qwen2vl", false, vec!["qwen2vl_merger", "qwen2.5vl_merger"]),
        ("qwen3vl", false, vec!["qwen3vl_merger"]),
        ("qwen3vlmoe", false, vec!["qwen3vl_merger"]),
    ] {
        let detection = detect_model_from_gguf_bytes(
            format!("{architecture}.gguf"),
            &metadata_gguf(&[
                ("general.architecture", FixtureValue::String(architecture)),
                ("clip.has_vision_encoder", FixtureValue::Bool(has_encoder)),
            ]),
        )
        .expect("detection");

        assert_eq!(detection.inspection.role, AssetRole::Model);
        assert_eq!(
            detection.inspection.compatible_vision_projector_types,
            expected
        );
        assert!(detection.inspection.vision_capable);
    }
}

#[test]
fn vision_encoder_required_architectures_stay_text_only_without_encoder_flag() {
    let detection = detect_model_from_gguf_bytes(
        "gemma3.gguf",
        &metadata_gguf(&[("general.architecture", FixtureValue::String("gemma3"))]),
    )
    .expect("detection");

    assert_eq!(detection.inspection.role, AssetRole::Model);
    assert!(!detection.inspection.vision_capable);
    assert!(detection
        .inspection
        .compatible_vision_projector_types
        .is_empty());
}

#[test]
fn unknown_architecture_is_model_without_vision_projector_types() {
    let detection = detect_model_from_gguf_bytes(
        "unknown.gguf",
        &metadata_gguf(&[("general.architecture", FixtureValue::String("llama"))]),
    )
    .expect("detection");

    assert_eq!(detection.inspection.role, AssetRole::Model);
    assert!(!detection.inspection.vision_capable);
    assert!(detection
        .inspection
        .compatible_vision_projector_types
        .is_empty());
}

#[test]
fn helper_predicates_cover_each_target_and_useful_metadata_branch() {
    assert!(has_useful_metadata(
        Some(&"model".to_string()),
        None,
        None,
        None,
        None,
        None
    ));
    assert!(has_useful_metadata(
        None,
        Some(&"llama".to_string()),
        None,
        None,
        None,
        None
    ));
    assert!(has_useful_metadata(None, None, Some(1), None, None, None));
    assert!(has_useful_metadata(
        None,
        None,
        None,
        Some(&"legacy".to_string()),
        None,
        None
    ));
    assert!(has_useful_metadata(
        None,
        None,
        None,
        None,
        Some(&"modern".to_string()),
        None
    ));
    assert!(has_useful_metadata(
        None,
        None,
        None,
        None,
        None,
        Some(false)
    ));
    assert!(!has_useful_metadata(None, None, None, None, None, None));

    assert!(is_target_key("general.type"));
    assert!(is_target_key("bert.pooling_type"));
    assert!(is_pooling_key("general.pooling_type"));
    assert!(!is_target_key("tokenizer.ggml.tokens"));
}

fn metadata_header(kv_count: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    push_u32(&mut bytes, GGUF_MAGIC);
    push_u32(&mut bytes, 3);
    push_u64(&mut bytes, 0);
    push_u64(&mut bytes, kv_count);
    bytes
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_string(bytes: &mut Vec<u8>, value: &str) {
    push_u64(bytes, u64::try_from(value.len()).expect("string length"));
    bytes.extend_from_slice(value.as_bytes());
}
