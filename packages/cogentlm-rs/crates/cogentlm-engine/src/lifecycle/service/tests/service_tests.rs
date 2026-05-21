//! Unit tests for the parent module.

use super::super::helpers::model_id_from_plan;
use super::super::*;
use crate::lifecycle::{
    model_entry_from_assets, AssetInspection, AssetRecord, AssetRole, ModelAssetKind, PairingPlan,
};
use std::fs;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "cogentlm-engine-service-{}-{}",
            name,
            now_unix_ms()
        ));
        fs::create_dir_all(&path).expect("temp dir");
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn model_id_is_stable_for_asset_order() {
    let left = PairingPlan {
        model_asset_ids: vec!["asset-b".to_string(), "asset-a".to_string()],
        projector_asset_id: Some("asset-c".to_string()),
        name: "model".to_string(),
        modality: ModelModality::Vision,
        status: ModelStatus::Ready,
        compatible_vision_projector_types: Vec::new(),
    };
    let right = PairingPlan {
        model_asset_ids: vec!["asset-a".to_string(), "asset-b".to_string()],
        projector_asset_id: Some("asset-c".to_string()),
        ..left.clone()
    };

    assert_eq!(model_id_from_plan(&left), model_id_from_plan(&right));
}

#[test]
fn service_installs_and_lists_text_asset() {
    let root = TempDir::new("install-list");
    let model = root.path.join("model.gguf");
    fs::write(&model, b"not a gguf").expect("model");

    let mut service = ModelService::local(root.path.join("store")).expect("service");
    let source = model_source_from_path(&model);
    let result = service.resolve_source(source).expect("resolved");

    let models = service.list();
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, result.entry_id);
    assert_eq!(models[0].status, ModelStatus::Ready);
    assert_eq!(models[0].bytes, 10);
}

#[test]
fn cached_local_asset_requires_matching_source_hash() {
    let root = TempDir::new("cache-hash");
    let model = root.path.join("model.gguf");
    fs::write(&model, b"first bytes").expect("model");

    let mut service = ModelService::local(root.path.join("store")).expect("service");
    let first = service
        .resolve_source(model_source_from_path(&model))
        .expect("first");
    let first_asset_id = service
        .registry
        .model(&first.entry_id)
        .expect("first model")
        .model_asset_ids[0]
        .clone();

    fs::write(&model, b"secondbytes").expect("same len replacement");
    let second = service
        .resolve_source(model_source_from_path(&model))
        .expect("second");
    let second_asset_id = service
        .registry
        .model(&second.entry_id)
        .expect("second model")
        .model_asset_ids[0]
        .clone();

    assert_ne!(first_asset_id, second_asset_id);
}

#[test]
fn service_rejects_unresolved_vision_model_on_load() {
    let root = TempDir::new("needs-projector");
    let mut service = ModelService::local(root.path.join("store")).expect("service");
    let plan = PairingPlan {
        model_asset_ids: vec!["asset-a".to_string()],
        projector_asset_id: None,
        name: "vision".to_string(),
        modality: ModelModality::Vision,
        status: ModelStatus::NeedsProjector,
        compatible_vision_projector_types: vec!["lfm2".to_string()],
    };
    let mut record = AssetRecord {
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
            compatible_vision_projector_types: vec!["lfm2".to_string()],
            provided_vision_projector_type: None,
        }),
    };
    service
        .registry
        .upsert_asset(record.clone())
        .expect("asset");
    let entry_id = model_id_from_plan(&plan);
    let entry = model_entry_from_assets(&entry_id, "vision", &plan);
    service.registry.insert_model(entry).expect("model");
    record.ref_count = 1;

    let error = service
        .load_installed(&entry_id, ModelLoadOptions::default())
        .expect_err("not ready");

    assert!(matches!(error, ModelError::InvalidModelPairing(_)));
}
