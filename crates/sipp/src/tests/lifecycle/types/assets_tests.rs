//! Tests the `lifecycle::types::assets` module in `sipp`.
//!
//! Covers lifecycle registry, storage, browser, service, and pairing behavior with temporary storage and pure fixtures instead of native runtime loading.

use super::*;

#[test]
fn inspection_schema_four_round_trips_runtime_and_audio_metadata() {
    let inspection = AssetInspection {
        version: AssetInspection::VERSION,
        role: AssetRole::Projector,
        architecture: Some("clip".to_string()),
        trained_context_size: Some(2048),
        vision_capable: true,
        audio_capable: true,
        audio_generation_capable: true,
        compatible_vision_projector_types: Vec::new(),
        compatible_audio_projector_types: Vec::new(),
        compatible_audio_generation_projector_types: Vec::new(),
        provided_vision_projector_type: Some("combined".to_string()),
        provided_audio_projector_type: Some("audio".to_string()),
        provided_audio_generation_projector_type: Some("audio-gen".to_string()),
    };
    let value = serde_json::to_value(&inspection).expect("inspection");

    assert_eq!(value["version"], 4);
    assert_eq!(value["trainedContextSize"], 2048);
    assert_eq!(value["audioCapable"], true);
    assert_eq!(
        serde_json::from_value::<AssetInspection>(value).expect("inspection"),
        inspection
    );
}

#[test]
fn inspection_schema_three_requires_audio_generation_pairing_metadata() {
    let error = serde_json::from_str::<AssetInspection>(
        r#"{"version":3,"role":"model","architecture":null,"visionCapable":false,"audioCapable":true,"compatibleVisionProjectorTypes":[],"providedVisionProjectorType":null}"#,
    )
    .expect_err("schema three requires audio pairing metadata");

    assert!(error
        .to_string()
        .contains("missing field `audioGenerationCapable`"));
}

#[test]
fn local_asset_source_requires_source_path() {
    let error = serde_json::from_str::<AssetSource>(r#"{"kind":"local"}"#)
        .expect_err("local source without path should be rejected");

    assert!(error.to_string().contains("missing field `path`"));
}

#[test]
fn legacy_local_asset_source_defaults_to_absolute_path() {
    let source = serde_json::from_str::<AssetSource>(
        r#"{"kind":"local","path":"model.gguf","modified_unix_ms":null}"#,
    )
    .expect("legacy local source");

    assert!(matches!(
        source,
        AssetSource::Local {
            anchor: LocalPathAnchor::Absolute,
            ..
        }
    ));
}
