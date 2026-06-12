//! Tests the `lifecycle::registry` module in `cogentlm`.
//!
//! Covers lifecycle registry, storage, browser, service, and pairing behavior with temporary storage and pure fixtures instead of native runtime loading.

use super::*;
use crate::lifecycle::test_support::{gguf_name, gguf_path, TempDir};
use crate::lifecycle::{AssetSource, ModelAssetKind, ModelModality, ModelStatus, PairingPlan};
use std::fs;

fn asset(id: &str) -> AssetRecord {
    AssetRecord {
        id: id.to_string(),
        kind: ModelAssetKind::Model,
        name: gguf_name(id),
        hash: id.trim_start_matches("asset-").to_string(),
        bytes: 4,
        storage_path: PathBuf::from("assets").join(id),
        source: AssetSource::Local {
            path: gguf_path(id),
            modified_unix_ms: None,
        },
        ref_count: 0,
        created_at_unix_ms: now_unix_ms(),
        inspection: None,
    }
}

fn model(id: &str, asset_id: &str) -> ModelEntry {
    ModelEntry {
        id: id.to_string(),
        name: id.to_string(),
        modality: ModelModality::Text,
        status: ModelStatus::Ready,
        model_asset_ids: vec![asset_id.to_string()],
        projector_asset_id: None,
        pairing: None,
        runtime_fingerprint: None,
        created_at_unix_ms: now_unix_ms(),
        updated_at_unix_ms: now_unix_ms(),
        last_loaded_at_unix_ms: None,
    }
}

#[test]
fn registry_persists_assets_and_models() {
    let root = TempDir::new("registry", "persist");
    let mut registry = ModelRegistry::local(root.path.clone()).expect("registry");
    registry.upsert_asset(asset("asset-a")).expect("asset");
    registry
        .insert_model(model("model-a", "asset-a"))
        .expect("model");
    registry.save().expect("save");

    let loaded = ModelRegistry::local(root.path.clone()).expect("reload");
    assert_eq!(loaded.manifest.assets["asset-a"].ref_count, 1);
    assert_eq!(loaded.manifest.models["model-a"].name, "model-a");
}

#[test]
fn registry_removes_model_and_returns_orphaned_assets() {
    let root = TempDir::new("registry", "remove");
    let mut registry = ModelRegistry::local(root.path.clone()).expect("registry");
    registry.upsert_asset(asset("asset-a")).expect("asset");
    registry
        .insert_model(model("model-a", "asset-a"))
        .expect("model");

    let removed = registry.remove_model("model-a").expect("remove");

    assert_eq!(removed.model.id, "model-a");
    assert_eq!(removed.orphaned_assets.len(), 1);
    assert!(registry.manifest.assets.is_empty());
    assert!(registry.manifest.models.is_empty());
}

#[test]
fn update_model_rebalances_asset_refcounts_when_assets_change() {
    let root = TempDir::new("registry", "update-refs");
    let mut registry = ModelRegistry::local(root.path.clone()).expect("registry");
    registry.upsert_asset(asset("asset-a")).expect("asset a");
    registry.upsert_asset(asset("asset-b")).expect("asset b");
    registry
        .insert_model(model("model-a", "asset-a"))
        .expect("model");

    registry
        .update_model("model-a", |entry| {
            entry.model_asset_ids = vec!["asset-b".to_string()];
        })
        .expect("update");

    assert_eq!(registry.manifest.assets["asset-a"].ref_count, 0);
    assert_eq!(registry.manifest.assets["asset-b"].ref_count, 1);
    assert_eq!(
        registry.manifest.models["model-a"].model_asset_ids,
        vec!["asset-b"]
    );
}

#[test]
fn insert_model_rejects_refcount_overflow() {
    let root = TempDir::new("registry", "refcount-overflow");
    let mut registry = ModelRegistry::local(root.path.clone()).expect("registry");
    let mut record = asset("asset-a");
    record.ref_count = u32::MAX;
    registry.upsert_asset(record).expect("asset");

    let error = registry
        .insert_model(model("model-a", "asset-a"))
        .expect_err("overflow");

    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message.contains("refcount overflow"))
    );
    assert!(!registry.manifest.models.contains_key("model-a"));
    assert_eq!(registry.manifest.assets["asset-a"].ref_count, u32::MAX);
}

#[test]
fn insert_model_rejects_projector_revision_overflow_without_mutating_manifest() {
    let root = TempDir::new("registry", "revision-overflow");
    let mut registry = ModelRegistry::local(root.path.clone()).expect("registry");
    registry.upsert_asset(asset("asset-a")).expect("asset");
    registry.manifest.projector_index_revision = u64::MAX;

    let error = registry
        .insert_model(model("model-a", "asset-a"))
        .expect_err("revision overflow");

    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message.contains("revision overflow"))
    );
    assert!(!registry.manifest.models.contains_key("model-a"));
    assert_eq!(registry.manifest.assets["asset-a"].ref_count, 0);
    assert_eq!(registry.manifest.projector_index_revision, u64::MAX);
}

#[test]
fn registry_reports_corrupt_manifest() {
    let root = TempDir::new("registry", "corrupt");
    fs::write(root.path.join("registry.json"), b"{\"version\":2}").expect("corrupt manifest");

    let error = ModelRegistry::local(root.path.clone()).expect_err("corrupt");
    assert!(matches!(error, ModelError::StorageCorrupt(_)));
}

#[test]
fn registry_reports_refcount_mismatch() {
    let root = TempDir::new("registry", "refcount-mismatch");
    let mut manifest = RegistryManifest::default();
    let mut asset = asset("asset-a");
    asset.ref_count = 0;
    manifest.assets.insert(asset.id.clone(), asset);
    manifest
        .models
        .insert("model-a".to_string(), model("model-a", "asset-a"));
    let bytes = serde_json::to_vec_pretty(&manifest).expect("manifest");
    fs::write(root.path.join("registry.json"), bytes).expect("manifest file");

    let error = ModelRegistry::local(root.path.clone()).expect_err("refcount mismatch");

    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message.contains("refcount mismatch"))
    );
}

#[test]
fn model_entry_helper_uses_pairing_plan() {
    let plan = PairingPlan {
        model_asset_ids: vec!["asset-a".to_string()],
        projector_asset_id: None,
        name: "planned".to_string(),
        modality: ModelModality::Text,
        status: ModelStatus::Ready,
        compatible_vision_projector_types: Vec::new(),
    };

    let entry = model_entry_from_assets("model-a", "model-a", &plan);

    assert_eq!(entry.model_asset_ids, vec!["asset-a"]);
    assert_eq!(entry.status, ModelStatus::Ready);
}
