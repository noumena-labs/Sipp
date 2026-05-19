//! Unit tests for the parent module.

use super::super::*;
use crate::lifecycle::AssetInspection;

fn model(id: &str, name: &str, vision_types: &[&str]) -> ClassifiedAsset {
    ClassifiedAsset {
        asset_id: id.to_string(),
        name: name.to_string(),
        inspection: AssetInspection {
            version: 1,
            role: AssetRole::Model,
            architecture: Some("test".to_string()),
            vision_capable: !vision_types.is_empty(),
            compatible_vision_projector_types: vision_types
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            provided_vision_projector_type: None,
        },
    }
}

fn projector(id: &str, name: &str, projector_type: Option<&str>) -> ClassifiedAsset {
    ClassifiedAsset {
        asset_id: id.to_string(),
        name: name.to_string(),
        inspection: AssetInspection {
            version: 1,
            role: AssetRole::Projector,
            architecture: Some("clip".to_string()),
            vision_capable: false,
            compatible_vision_projector_types: Vec::new(),
            provided_vision_projector_type: projector_type.map(str::to_string),
        },
    }
}

#[test]
fn resolves_text_model_as_ready() {
    let plan = PairingResolver::resolve(&[model("asset-model", "base.gguf", &[])]).expect("plan");

    assert_eq!(plan.modality, ModelModality::Text);
    assert_eq!(plan.status, ModelStatus::Ready);
    assert_eq!(plan.projector_asset_id, None);
}

#[test]
fn resolves_vision_base_as_needing_projector() {
    let plan =
        PairingResolver::resolve(&[model("asset-model", "base.gguf", &["lfm2"])]).expect("plan");

    assert_eq!(plan.modality, ModelModality::Vision);
    assert_eq!(plan.status, ModelStatus::NeedsProjector);
    assert_eq!(plan.compatible_vision_projector_types, vec!["lfm2"]);
}

#[test]
fn accepts_explicit_compatible_projector() {
    let base = model("asset-model", "base.gguf", &["lfm2"]);
    let mmproj = projector("asset-projector", "mmproj.gguf", Some("lfm2"));

    let plan = PairingResolver::resolve_explicit(&[base, mmproj], "asset-projector").expect("plan");

    assert_eq!(plan.modality, ModelModality::Vision);
    assert_eq!(plan.status, ModelStatus::Ready);
    assert_eq!(plan.projector_asset_id, Some("asset-projector".to_string()));
}

#[test]
fn accepts_single_implicit_compatible_projector() {
    let base = model("asset-model", "base.gguf", &["lfm2"]);
    let mmproj = projector("asset-projector", "mmproj.gguf", Some("lfm2"));

    let plan = PairingResolver::resolve(&[base, mmproj]).expect("plan");

    assert_eq!(plan.modality, ModelModality::Vision);
    assert_eq!(plan.status, ModelStatus::Ready);
    assert_eq!(plan.projector_asset_id, Some("asset-projector".to_string()));
}

#[test]
fn rejects_explicit_incompatible_projector() {
    let base = model("asset-model", "base.gguf", &["lfm2"]);
    let mmproj = projector("asset-projector", "bad-mmproj.gguf", Some("other"));

    let error = PairingResolver::resolve_explicit(&[base, mmproj], "asset-projector")
        .expect_err("pairing error");

    assert!(matches!(error, ModelError::InvalidModelPairing(_)));
}

#[test]
fn rejects_explicit_projector_for_text_model() {
    let base = model("asset-model", "base.gguf", &[]);
    let mmproj = projector("asset-projector", "mmproj.gguf", Some("lfm2"));

    let error = PairingResolver::resolve_explicit(&[base, mmproj], "asset-projector")
        .expect_err("pairing error");

    assert!(matches!(error, ModelError::InvalidModelPairing(_)));
}

#[test]
fn rejects_implicit_projector_for_text_model() {
    let base = model("asset-model", "base.gguf", &[]);
    let mmproj = projector("asset-projector", "mmproj.gguf", Some("lfm2"));

    let error = PairingResolver::resolve(&[base, mmproj]).expect_err("pairing error");

    assert!(matches!(error, ModelError::InvalidModelPairing(_)));
}

#[test]
fn rejects_multiple_implicit_projectors() {
    let base = model("asset-model", "base.gguf", &["lfm2"]);
    let first = projector("asset-projector-a", "a.gguf", Some("lfm2"));
    let second = projector("asset-projector-b", "b.gguf", Some("lfm2"));

    let error = PairingResolver::resolve(&[base, first, second]).expect_err("pairing error");

    assert!(
        matches!(error, ModelError::InvalidModelPairing(message) if message.ends_with("a.gguf, b.gguf"))
    );
}

#[test]
fn asset_name_joining_presizes_exact_message_capacity() {
    let first = projector("asset-projector-a", "a.gguf", Some("lfm2"));
    let second = projector("asset-projector-b", "b.gguf", Some("lfm2"));
    let files = vec![&first, &second];

    let names = join_asset_names(&files);

    assert_eq!(names, "a.gguf, b.gguf");
    assert_eq!(names.capacity(), names.len());
    assert_eq!(joined_asset_names_capacity(&files), Some(names.len()));
}

#[test]
fn rejects_shards_with_conflicting_projector_types() {
    let first = model("asset-a", "a.gguf", &["lfm2"]);
    let second = model("asset-b", "b.gguf", &["qwen3vl_merger"]);

    let error = PairingResolver::resolve(&[first, second]).expect_err("source error");

    assert!(matches!(error, ModelError::InvalidModelSource(_)));
}
