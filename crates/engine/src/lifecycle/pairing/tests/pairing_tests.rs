//! Unit tests for the parent module.

use super::super::*;
use crate::lifecycle::test_support::{some_string, strings};
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
            compatible_vision_projector_types: strings(vision_types),
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

fn unknown(id: &str, name: &str) -> ClassifiedAsset {
    ClassifiedAsset {
        asset_id: id.to_string(),
        name: name.to_string(),
        inspection: AssetInspection::unknown(),
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
    assert_eq!(plan.projector_asset_id, some_string("asset-projector"));
}

#[test]
fn accepts_single_implicit_compatible_projector() {
    let base = model("asset-model", "base.gguf", &["lfm2"]);
    let mmproj = projector("asset-projector", "mmproj.gguf", Some("lfm2"));

    let plan = PairingResolver::resolve(&[base, mmproj]).expect("plan");

    assert_eq!(plan.modality, ModelModality::Vision);
    assert_eq!(plan.status, ModelStatus::Ready);
    assert_eq!(plan.projector_asset_id, some_string("asset-projector"));
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
fn accepts_explicit_projector_when_base_metadata_is_inconclusive() {
    let base = model("asset-model", "base.gguf", &[]);
    let mmproj = projector("asset-projector", "mmproj.gguf", Some("lfm2"));

    let plan = PairingResolver::resolve_explicit(&[base, mmproj], "asset-projector")
        .expect("explicit projector override");

    assert_eq!(plan.modality, ModelModality::Vision);
    assert_eq!(plan.status, ModelStatus::Ready);
    assert_eq!(plan.projector_asset_id, some_string("asset-projector"));
}

#[test]
fn rejects_explicit_projector_id_when_asset_is_not_a_projector() {
    let base = model("asset-model", "base.gguf", &[]);
    let named_projector = unknown("asset-projector", "mmproj-LFM2-VL-1.6B-f16.gguf");

    let error = PairingResolver::resolve_explicit(&[base, named_projector], "asset-projector")
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
fn rejects_shards_with_conflicting_projector_types() {
    let first = model("asset-a", "a.gguf", &["lfm2"]);
    let second = model("asset-b", "b.gguf", &["qwen3vl_merger"]);

    let error = PairingResolver::resolve(&[first, second]).expect_err("source error");

    assert!(matches!(error, ModelError::InvalidModelSource(_)));
}
