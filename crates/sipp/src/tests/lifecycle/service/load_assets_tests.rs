//! Tests the `lifecycle::service::load_assets` module in `sipp`.
//!
//! Covers lifecycle registry, storage, browser, service, and pairing behavior with temporary storage and pure fixtures instead of native runtime loading.

use super::*;
use crate::lifecycle::storage::now_unix_ms;
use crate::lifecycle::test_support::{gguf_name, strings, TempDir};
use crate::lifecycle::{
    AssetRecord, AssetRole, AssetSource, LocalPathAnchor, ModelAssetKind, ModelEntry,
    ModelModality, ModelStatus, ModelStore,
};
use futures::executor::block_on;
use std::fs;

fn asset_record(id: &str, path: PathBuf) -> AssetRecord {
    AssetRecord {
        id: id.to_string(),
        kind: ModelAssetKind::Model,
        name: gguf_name(id),
        hash: id.to_string(),
        bytes: 1,
        storage_path: path.clone(),
        source: AssetSource::Local {
            path,
            anchor: LocalPathAnchor::Absolute,
            modified_unix_ms: None,
        },
        ref_count: 1,
        created_at_unix_ms: now_unix_ms(),
        inspection: Some(crate::lifecycle::AssetInspection {
            version: crate::lifecycle::AssetInspection::VERSION,
            role: AssetRole::Model,
            architecture: None,
            trained_context_size: Some(4096),
            vision_capable: false,
            audio_capable: false,
            audio_generation_capable: false,
            compatible_vision_projector_types: Vec::new(),
            compatible_audio_projector_types: Vec::new(),
            compatible_audio_generation_projector_types: Vec::new(),
            provided_vision_projector_type: None,
            provided_audio_projector_type: None,
            provided_audio_generation_projector_type: None,
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
    let store = ModelStore::local(root.path.join("store")).expect("store");
    let entry = model_entry(strings(&["missing"]));

    let error = block_on(store.state.lock())
        .resolve_load_asset_paths(&entry)
        .expect_err("missing asset");

    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message.contains("missing asset"))
    );
}

#[test]
fn resolve_load_asset_paths_returns_local_source_path() {
    let root = TempDir::new("load-assets", "load-asset-path");
    let store = ModelStore::local(root.path.join("store")).expect("store");
    let source_path = root.path.join("asset-a.gguf");
    fs::write(&source_path, [0_u8]).expect("asset bytes");
    let record = asset_record("asset-a", source_path.clone());
    let entry = model_entry(strings(&["asset-a"]));
    let mut state = block_on(store.state.lock());
    state.registry.upsert_asset(record).expect("asset");
    let paths = state
        .resolve_load_asset_paths(&entry)
        .expect("load asset paths");

    assert_eq!(paths.model_path, source_path);
    assert!(paths.projector_path.is_none());
}
