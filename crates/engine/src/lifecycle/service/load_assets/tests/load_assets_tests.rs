//! Unit tests for the parent module.

use super::super::*;
use crate::lifecycle::storage::now_unix_ms;
use crate::lifecycle::test_support::{gguf_name, gguf_path, strings, TempDir};
use crate::lifecycle::{
    AssetRecord, AssetRole, AssetSource, ModelAssetKind, ModelEntry, ModelModality, ModelStatus,
};
use std::fs;

fn asset_record(id: &str, storage_path: impl Into<PathBuf>) -> AssetRecord {
    AssetRecord {
        id: id.to_string(),
        kind: ModelAssetKind::Model,
        name: gguf_name(id),
        hash: id.to_string(),
        bytes: 1,
        storage_path: storage_path.into(),
        source: AssetSource::Local {
            path: gguf_path(id),
            modified_unix_ms: None,
        },
        ref_count: 1,
        created_at_unix_ms: now_unix_ms(),
        inspection: Some(crate::lifecycle::AssetInspection {
            version: 1,
            role: AssetRole::Model,
            architecture: None,
            vision_capable: false,
            compatible_vision_projector_types: Vec::new(),
            provided_vision_projector_type: None,
        }),
    }
}

fn model_entry(asset_ids: Vec<String>) -> ModelEntry {
    ModelEntry {
        id: "model".to_string(),
        name: "model".to_string(),
        modality: ModelModality::Text,
        status: ModelStatus::Ready,
        model_asset_ids: asset_ids,
        projector_asset_id: None,
        pairing: None,
        runtime_fingerprint: None,
        last_loaded_at_unix_ms: None,
        created_at_unix_ms: now_unix_ms(),
        updated_at_unix_ms: now_unix_ms(),
    }
}

#[test]
fn resolve_load_asset_paths_rejects_missing_model_asset() {
    let root = TempDir::new("load-assets", "missing-load-asset");
    let service = ModelService::local(root.path.join("store")).expect("service");
    let entry = model_entry(strings(&["missing"]));

    let error = service
        .resolve_load_asset_paths(&entry)
        .expect_err("missing asset");

    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message.contains("missing asset"))
    );
}

#[test]
fn resolve_load_asset_paths_returns_storage_path() {
    let root = TempDir::new("load-assets", "load-asset-path");
    let mut service = ModelService::local(root.path.join("store")).expect("service");
    let record = asset_record("asset-a", PathBuf::from("assets/asset-a.gguf"));
    let stored_path = root.path.join("store").join(&record.storage_path);
    fs::create_dir_all(stored_path.parent().expect("asset parent")).expect("asset dir");
    fs::write(&stored_path, [0_u8]).expect("asset bytes");
    service.registry.upsert_asset(record).expect("asset");
    let entry = model_entry(strings(&["asset-a"]));

    let paths = service
        .resolve_load_asset_paths(&entry)
        .expect("load asset paths");

    assert!(paths.model_path.ends_with("assets/asset-a.gguf"));
    assert!(paths.projector_path.is_none());
}
