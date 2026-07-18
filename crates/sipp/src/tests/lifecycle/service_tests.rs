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
fn service_installs_and_lists_text_asset() {
    let root = TempDir::new("service", "install-list");
    let model = root.path.join("model.gguf");
    fs::write(&model, b"not a gguf").expect("model");

    let store = ModelStore::local(root.path.join("store")).expect("store");
    let installed = block_on(store.install_files([model])).expect("installed");
    let state = block_on(store.state.lock());
    let model = state
        .registry
        .manifest
        .models
        .get(&installed.id)
        .expect("installed model");
    assert_eq!(model.status, ModelStatus::Ready);
}

#[test]
fn store_rejects_removing_a_model_in_use() {
    let root = TempDir::new("service", "remove-in-use");
    let model_path = root.path.join("model.gguf");
    fs::write(&model_path, b"not a gguf").expect("model");

    let store = ModelStore::local(root.path.join("store")).expect("store");
    let model = block_on(store.install_files([model_path])).expect("installed");
    block_on(store.replace_usage(None, Some(&model.id)));

    let error = block_on(store.remove(&model.id)).expect_err("model is in use");
    assert!(matches!(error, ModelError::ModelInUse(id) if id == model.id));

    block_on(store.replace_usage(Some(&model.id), None));
    block_on(store.remove(&model.id)).expect("removed");
}

#[test]
fn cached_local_asset_requires_matching_source_hash() {
    let root = TempDir::new("service", "cache-hash");
    let model = root.path.join("model.gguf");
    fs::write(&model, b"first bytes").expect("model");

    let store = ModelStore::local(root.path.join("store")).expect("store");
    let first = block_on(store.install_files([model.clone()])).expect("first");
    let first_asset_id = block_on(store.state.lock())
        .registry
        .manifest
        .models
        .get(&first.id)
        .expect("first model")
        .model_asset_ids[0]
        .clone();

    fs::write(&model, b"secondbytes").expect("same len replacement");
    let second = block_on(store.install_files([model])).expect("second");
    let second_asset_id = block_on(store.state.lock())
        .registry
        .manifest
        .models
        .get(&second.id)
        .expect("second model")
        .model_asset_ids[0]
        .clone();

    assert_ne!(first_asset_id, second_asset_id);
}

#[test]
fn service_rejects_unresolved_vision_model_on_load() {
    let root = TempDir::new("service", "needs-projector");
    let store = ModelStore::local(root.path.join("store")).expect("store");
    let plan = vision_plan();
    let record = AssetRecord {
        id: "asset-a".to_string(),
        kind: ModelAssetKind::Model,
        name: "vision.gguf".to_string(),
        hash: "a".to_string(),
        bytes: 1,
        storage_path: PathBuf::from("assets/asset-a"),
        source: AssetSource::Local {
            path: PathBuf::from("vision.gguf"),
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
