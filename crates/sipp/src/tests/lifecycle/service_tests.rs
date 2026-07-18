//! Tests the `lifecycle::service` module in `sipp`.
//!
//! Covers registry, storage, service, and pairing behavior with temporary
//! storage and pure fixtures instead of native runtime loading.

use super::helpers::model_id_from_plan;
use super::*;
use crate::lifecycle::test_support::{some_string, strings, TempDir};
use crate::lifecycle::{
    model_entry_from_assets, AssetInspection, AssetRecord, AssetRole, AssetSource, ModelAssetKind,
    ModelModality, PairingPlan,
};
use futures::executor::block_on;
use std::{fs, path::PathBuf};

fn vision_plan() -> PairingPlan {
    PairingPlan {
        model_asset_ids: strings(&["asset-a"]),
        projector_asset_id: None,
        name: "vision".to_string(),
        modality: ModelModality::Vision,
        status: ModelStatus::NeedsProjector,
        compatible_vision_projector_types: strings(&["lfm2"]),
    }
}

#[test]
fn model_id_is_stable_for_asset_order() {
    let left = PairingPlan {
        model_asset_ids: strings(&["asset-b", "asset-a"]),
        projector_asset_id: some_string("asset-c"),
        name: "model".to_string(),
        modality: ModelModality::Vision,
        status: ModelStatus::Ready,
        compatible_vision_projector_types: Vec::new(),
    };
    let right = PairingPlan {
        model_asset_ids: strings(&["asset-a", "asset-b"]),
        projector_asset_id: some_string("asset-c"),
        ..left.clone()
    };

    assert_eq!(model_id_from_plan(&left), model_id_from_plan(&right));
}

#[test]
#[ignore = "requires repo-root t5-small-f16.gguf fixture; run model-backed checks through xtask test run --suite model-smoke"]
fn t5_encoder_decoder_fixture_is_available() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("t5-small-f16.gguf");
    let metadata = crate::shard::inspect_gguf_metadata_path(&path)
        .expect("repo-root t5-small-f16.gguf metadata")
        .expect("repo-root t5-small-f16.gguf is GGUF");

    assert_eq!(metadata.general_architecture.as_deref(), Some("t5"));
}

#[test]
fn service_adds_and_lists_local_model_without_copying_it() {
    let root = TempDir::new("service", "add-list");
    let model_path = root.path.join("model.gguf");
    fs::write(&model_path, b"not a gguf").expect("model");

    let store_root = root.path.join("store");
    let store = ModelStore::local(&store_root).expect("store");
    let added = block_on(store.add([&model_path])).expect("added");
    let listed = block_on(store.list()).expect("list");
    let state = block_on(store.state.lock());
    let model = state
        .registry
        .manifest
        .models
        .get(&added.id)
        .expect("added model");

    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].id, added.id);
    assert_eq!(model.status, ModelStatus::Ready);
    assert_eq!(
        fs::read_dir(store_root.join("assets"))
            .expect("assets directory")
            .count(),
        0
    );
}

#[test]
fn store_rejects_removing_a_model_in_use() {
    let root = TempDir::new("service", "remove-in-use");
    let model_path = root.path.join("model.gguf");
    fs::write(&model_path, b"not a gguf").expect("model");

    let store = ModelStore::local(root.path.join("store")).expect("store");
    let model = block_on(store.add([&model_path])).expect("added");
    block_on(store.replace_usage(None, Some(&model.id)));

    let error = block_on(store.remove(&model.id)).expect_err("model is in use");
    assert!(matches!(error, ModelError::ModelInUse(id) if id == model.id));

    block_on(store.replace_usage(Some(&model.id), None));
    block_on(store.remove(&model.id)).expect("removed");
    assert!(model_path.exists());
}

#[test]
fn adding_a_changed_local_source_replaces_its_registry_entry() {
    let root = TempDir::new("service", "replace-local");
    let model = root.path.join("model.gguf");
    fs::write(&model, b"first bytes").expect("model");

    let store = ModelStore::local(root.path.join("store")).expect("store");
    let first = block_on(store.add([&model])).expect("first");
    let first_asset_id = block_on(store.state.lock())
        .registry
        .manifest
        .models
        .get(&first.id)
        .expect("first model")
        .model_asset_ids[0]
        .clone();

    fs::write(&model, b"secondbytes").expect("same len replacement");
    let second = block_on(store.add([&model])).expect("second");
    let state = block_on(store.state.lock());
    let second_asset_id = state
        .registry
        .manifest
        .models
        .get(&second.id)
        .expect("second model")
        .model_asset_ids[0]
        .clone();

    assert_ne!(first_asset_id, second_asset_id);
    assert_eq!(state.registry.manifest.models.len(), 1);
    assert_eq!(state.registry.manifest.assets.len(), 1);
}

#[test]
fn listing_prunes_a_deleted_local_source() {
    let root = TempDir::new("service", "prune-local");
    let model = root.path.join("model.gguf");
    fs::write(&model, b"model bytes").expect("model");

    let store = ModelStore::local(root.path.join("store")).expect("store");
    block_on(store.add([&model])).expect("add");
    fs::remove_file(model).expect("delete source");

    assert!(block_on(store.list()).expect("list").is_empty());
    let state = block_on(store.state.lock());
    assert!(state.registry.manifest.models.is_empty());
    assert!(state.registry.manifest.assets.is_empty());
}

#[test]
fn service_rejects_unresolved_vision_model_on_load() {
    let root = TempDir::new("service", "needs-projector");
    let store = ModelStore::local(root.path.join("store")).expect("store");
    let model_path = root.path.join("vision.gguf");
    fs::write(&model_path, b"v").expect("model");
    let plan = vision_plan();
    let record = AssetRecord {
        id: "asset-a".to_string(),
        kind: ModelAssetKind::Model,
        name: "vision.gguf".to_string(),
        hash: "a".to_string(),
        bytes: 1,
        storage_path: model_path.clone(),
        source: AssetSource::Local {
            path: model_path,
            modified_unix_ms: None,
        },
        ref_count: 0,
        created_at_unix_ms: now_unix_ms(),
        inspection: Some(AssetInspection {
            version: 1,
            role: AssetRole::Model,
            architecture: Some("lfm2".to_string()),
            vision_capable: true,
            compatible_vision_projector_types: strings(&["lfm2"]),
            provided_vision_projector_type: None,
        }),
    };
    let entry_id = model_id_from_plan(&plan);
    {
        let mut state = block_on(store.state.lock());
        state.registry.upsert_asset(record).expect("asset");
        let entry = model_entry_from_assets(&entry_id, "vision", &plan);
        state.registry.insert_model(entry).expect("model");
    }

    let error = match block_on(store.load_engine(&entry_id, ModelLoadOptions::default())) {
        Ok(_) => panic!("unresolved vision model loaded"),
        Err(error) => error,
    };

    assert!(matches!(error, ModelError::InvalidModelPairing(_)));
}
