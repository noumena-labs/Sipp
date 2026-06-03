//! Tests the `lifecycle::registry::refs` module in `cogentlm-engine`.
//!
//! Covers manifest reference validation and refcount rebalancing with pure
//! in-memory registry manifests and no filesystem access.

use std::path::PathBuf;

use crate::lifecycle::{
    AssetRecord, AssetSource, ModelAssetKind, ModelModality, ModelStatus, REGISTRY_MANIFEST_VERSION,
};

use super::*;

fn asset(id: &str, ref_count: u32) -> AssetRecord {
    AssetRecord {
        id: id.to_string(),
        kind: ModelAssetKind::Model,
        name: format!("{id}.gguf"),
        hash: id.to_string(),
        bytes: 1,
        storage_path: PathBuf::from("assets").join(id),
        source: AssetSource::Local {
            path: PathBuf::from(format!("{id}.gguf")),
            modified_unix_ms: None,
        },
        ref_count,
        created_at_unix_ms: 0,
        inspection: None,
    }
}

fn model(id: &str, model_asset_ids: &[&str], projector_asset_id: Option<&str>) -> ModelEntry {
    ModelEntry {
        id: id.to_string(),
        name: id.to_string(),
        modality: ModelModality::Text,
        status: ModelStatus::Ready,
        model_asset_ids: model_asset_ids
            .iter()
            .map(|value| (*value).to_string())
            .collect(),
        projector_asset_id: projector_asset_id.map(str::to_string),
        pairing: None,
        runtime_fingerprint: None,
        created_at_unix_ms: 0,
        updated_at_unix_ms: 0,
        last_loaded_at_unix_ms: None,
    }
}

fn manifest_with_assets(refs: &[(&str, u32)]) -> RegistryManifest {
    let mut manifest = RegistryManifest {
        version: REGISTRY_MANIFEST_VERSION,
        projector_index_revision: 0,
        ..RegistryManifest::default()
    };
    for (id, ref_count) in refs {
        manifest
            .assets
            .insert((*id).to_string(), asset(id, *ref_count));
    }
    manifest
}

#[test]
fn referenced_asset_ids_sorts_deduplicates_and_includes_projector() {
    let model = model(
        "model-a",
        &["asset-b", "asset-a", "asset-a"],
        Some("asset-c"),
    );

    assert_eq!(
        referenced_asset_ids(&model),
        ["asset-a", "asset-b", "asset-c"]
    );
}

#[test]
fn validate_manifest_accepts_matching_refcounts() {
    let mut manifest = manifest_with_assets(&[("asset-a", 1), ("asset-b", 1)]);
    manifest.models.insert(
        "model-a".to_string(),
        model("model-a", &["asset-a"], Some("asset-b")),
    );

    validate_manifest(&manifest).expect("valid manifest");
}

#[test]
fn validate_manifest_rejects_missing_model_asset() {
    let mut manifest = manifest_with_assets(&[]);
    manifest
        .models
        .insert("model-a".to_string(), model("model-a", &["asset-a"], None));

    let error = validate_manifest(&manifest).expect_err("missing asset");

    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message.contains("model model-a references missing asset asset-a"))
    );
}

#[test]
fn rebalance_refs_moves_counts_and_bumps_revision() {
    let mut manifest = manifest_with_assets(&[("asset-a", 1), ("asset-b", 0)]);

    rebalance_refs(
        &mut manifest,
        &["asset-a".to_string()],
        &["asset-b".to_string()],
    )
    .expect("rebalance");

    assert_eq!(manifest.assets["asset-a"].ref_count, 0);
    assert_eq!(manifest.assets["asset-b"].ref_count, 1);
    assert_eq!(manifest.projector_index_revision, 2);
}

#[test]
fn rebalance_refs_noop_preserves_revision() {
    let mut manifest = manifest_with_assets(&[("asset-a", 1)]);

    rebalance_refs(
        &mut manifest,
        &["asset-a".to_string()],
        &["asset-a".to_string()],
    )
    .expect("rebalance");

    assert_eq!(manifest.assets["asset-a"].ref_count, 1);
    assert_eq!(manifest.projector_index_revision, 0);
}

#[test]
fn ref_adjustments_report_missing_and_overflowing_assets() {
    let mut missing = manifest_with_assets(&[]);
    let error =
        increment_refs(&mut missing, vec!["asset-a".to_string()]).expect_err("missing asset");
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "model references missing asset asset-a")
    );

    let mut overflowing = manifest_with_assets(&[("asset-a", u32::MAX)]);
    let error =
        increment_refs(&mut overflowing, vec!["asset-a".to_string()]).expect_err("overflow");
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message.contains("refcount overflow"))
    );
}
