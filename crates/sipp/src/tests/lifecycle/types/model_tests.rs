//! Tests the `lifecycle::types::model` module in `sipp`.
//!
//! Covers model lifecycle value contracts, serde wire names, and default
//! manifest boundaries with pure value fixtures.

use std::path::PathBuf;

use serde_json::json;

use super::*;

#[test]
fn model_enum_as_str_values_match_wire_names() {
    assert_eq!(ModelModality::Text.as_str(), "text");
    assert_eq!(ModelModality::Vision.as_str(), "vision");
    assert_eq!(ModelStatus::Ready.as_str(), "ready");
    assert_eq!(ModelStatus::NeedsProjector.as_str(), "needs_projector");
    assert_eq!(ModelStatus::Broken.as_str(), "broken");
    assert_eq!(ModelSourceKind::Local.as_str(), "local");
    assert_eq!(ModelSourceKind::Remote.as_str(), "remote");
}

#[test]
fn model_source_variants_use_snake_case_tags() {
    let source: ModelSource = serde_json::from_value(json!({
        "kind": "local",
        "model_paths": ["a.gguf", "b.gguf"],
        "projector_path": "mmproj.gguf"
    }))
    .expect("model source");

    assert!(matches!(
        source,
        ModelSource::Local {
            model_paths,
            projector_path: Some(projector_path)
        }
        if model_paths == [PathBuf::from("a.gguf"), PathBuf::from("b.gguf")]
            && projector_path == PathBuf::from("mmproj.gguf")
    ));

    let installed: ModelSource =
        serde_json::from_value(json!({ "kind": "installed", "model_id": "model-a" }))
            .expect("installed source");
    assert_eq!(
        installed,
        ModelSource::Installed {
            model_id: "model-a".to_string()
        }
    );

    let remote = ModelSource::Remote {
        model_urls: vec!["https://example.test/model.gguf".to_string()],
        projector_url: None,
    };
    assert_eq!(
        serde_json::to_value(remote).expect("remote source"),
        json!({
            "kind": "remote",
            "model_urls": ["https://example.test/model.gguf"],
            "projector_url": null
        })
    );
}

#[test]
fn model_info_entry_pairing_and_plan_use_camel_case_contracts() {
    let info = ModelInfo {
        id: "model-a".to_string(),
        name: "Model A".to_string(),
        modality: ModelModality::Vision,
        status: ModelStatus::NeedsProjector,
        source: ModelSourceKind::Local,
        bytes: 42,
        loaded: true,
        chat_template: Some("{{ messages }}".to_string()),
        bos_text: "<s>".to_string(),
        eos_text: "</s>".to_string(),
        media_marker: Some("<image>".to_string()),
        created_at_unix_ms: 1,
        updated_at_unix_ms: 2,
    };
    let value = serde_json::to_value(&info).expect("model info");
    assert_eq!(value["chatTemplate"], "{{ messages }}");
    assert_eq!(value["bosText"], "<s>");
    assert_eq!(value["eosText"], "</s>");
    assert_eq!(value["mediaMarker"], "<image>");
    assert_eq!(value["createdAtUnixMs"], 1);
    assert_eq!(
        serde_json::from_value::<ModelInfo>(value).expect("model info"),
        info
    );

    let pairing = ModelPairing {
        state: ModelPairingState::Unresolved,
        checked_projector_index_revision: 3,
        compatible_vision_projector_types: vec!["clip".to_string()],
        reason: Some(ModelPairingReason::MultipleMatches),
        updated_at_unix_ms: 4,
    };
    let entry = ModelEntry {
        id: "entry".to_string(),
        name: "Entry".to_string(),
        modality: ModelModality::Vision,
        status: ModelStatus::Ready,
        model_asset_ids: vec!["asset-a".to_string()],
        projector_asset_id: Some("projector-a".to_string()),
        pairing: Some(pairing.clone()),
        runtime_fingerprint: Some("fingerprint".to_string()),
        created_at_unix_ms: 5,
        updated_at_unix_ms: 6,
        last_loaded_at_unix_ms: Some(7),
    };
    let entry_value = serde_json::to_value(&entry).expect("entry");
    assert_eq!(entry_value["modelAssetIds"], json!(["asset-a"]));
    assert_eq!(entry_value["projectorAssetId"], "projector-a");
    assert_eq!(entry_value["runtimeFingerprint"], "fingerprint");
    assert_eq!(entry_value["lastLoadedAtUnixMs"], 7);
    assert_eq!(
        serde_json::from_value::<ModelEntry>(entry_value).expect("entry"),
        entry
    );

    let plan = PairingPlan {
        model_asset_ids: vec!["model".to_string()],
        projector_asset_id: None,
        name: "Plan".to_string(),
        modality: ModelModality::Text,
        status: ModelStatus::Broken,
        compatible_vision_projector_types: Vec::new(),
    };
    let plan_value = serde_json::to_value(&plan).expect("plan");
    assert_eq!(plan_value["projectorAssetId"], serde_json::Value::Null);
    assert_eq!(plan_value["compatibleVisionProjectorTypes"], json!([]));
    assert_eq!(
        serde_json::from_value::<PairingPlan>(plan_value).expect("plan"),
        plan
    );

    assert_eq!(pairing.state, ModelPairingState::Unresolved);
}

#[test]
fn registry_manifest_default_uses_current_version_and_empty_maps() {
    let manifest = RegistryManifest::default();

    assert_eq!(manifest.version, REGISTRY_MANIFEST_VERSION);
    assert_eq!(manifest.projector_index_revision, 0);
    assert!(manifest.assets.is_empty());
    assert!(manifest.models.is_empty());
}

#[test]
fn pairing_reason_uses_screaming_snake_case_wire_names() {
    let value = serde_json::to_value(ModelPairingReason::BaseNotVision).expect("reason");

    assert_eq!(value, "BASE_NOT_VISION");
    assert_eq!(
        serde_json::from_value::<ModelPairingReason>(json!("MISSING_METADATA")).expect("reason"),
        ModelPairingReason::MissingMetadata
    );
}
