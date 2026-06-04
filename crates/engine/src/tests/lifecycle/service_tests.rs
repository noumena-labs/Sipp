//! Tests the `lifecycle::service` module in `cogentlm-engine`.
//!
//! Covers lifecycle registry, storage, browser, service, and pairing behavior with temporary storage and pure fixtures instead of native runtime loading.

use super::helpers::model_id_from_plan;
use super::*;
use crate::engine::protocol::EngineStatus;
use crate::engine::{ChatMessage, ChatRequest, ChatRole, EmbedOptions, EmbedRequest, QueryRequest};
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
    let metadata = cogentlm_shard::inspect_gguf_metadata_path(&path)
        .expect("repo-root t5-small-f16.gguf metadata")
        .expect("repo-root t5-small-f16.gguf is GGUF");

    assert_eq!(metadata.general_architecture.as_deref(), Some("t5"));
}

#[test]
fn service_installs_and_lists_text_asset() {
    let root = TempDir::new("service", "install-list");
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
    let root = TempDir::new("service", "cache-hash");
    let model = root.path.join("model.gguf");
    fs::write(&model, b"first bytes").expect("model");

    let mut service = ModelService::local(root.path.join("store")).expect("service");
    let first = service
        .resolve_source(model_source_from_path(&model))
        .expect("first");
    let first_asset_id = service
        .registry
        .manifest
        .models
        .get(&first.entry_id)
        .expect("first model")
        .model_asset_ids[0]
        .clone();

    fs::write(&model, b"secondbytes").expect("same len replacement");
    let second = service
        .resolve_source(model_source_from_path(&model))
        .expect("second");
    let second_asset_id = service
        .registry
        .manifest
        .models
        .get(&second.entry_id)
        .expect("second model")
        .model_asset_ids[0]
        .clone();

    assert_ne!(first_asset_id, second_asset_id);
}

#[test]
fn service_rejects_unresolved_vision_model_on_load() {
    let root = TempDir::new("service", "needs-projector");
    let mut service = ModelService::local(root.path.join("store")).expect("service");
    let plan = vision_plan();
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
            compatible_vision_projector_types: strings(&["lfm2"]),
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

    let error = block_on(service.load(
        ModelSource::Installed {
            id: entry_id.clone(),
        },
        ModelLoadOptions::default(),
    ))
    .expect_err("not ready");

    assert!(matches!(error, ModelError::InvalidModelPairing(_)));
}

#[test]
fn unloaded_service_reports_idle_and_rejects_runtime_facades() {
    let root = TempDir::new("service", "unloaded-facades");
    let mut service = ModelService::local(root.path.join("store")).expect("service");

    let state = block_on(service.state()).expect("idle state");
    assert_eq!(state.status, EngineStatus::Idle);
    assert!(state.model.is_none());
    block_on(service.unload()).expect("unload without current model is a no-op");

    let query_error = service
        .query(QueryRequest::new("hello"))
        .err()
        .expect("query without a loaded model");
    let chat_error = service
        .chat(ChatRequest::new(vec![ChatMessage::new(
            ChatRole::User,
            "hello",
        )]))
        .err()
        .expect("chat without a loaded model");
    let embed_error = service
        .embed(EmbedRequest {
            input: "hello".to_string(),
            options: EmbedOptions::default(),
        })
        .err()
        .expect("embed without a loaded model");
    let subscribe_error = service
        .subscribe_events()
        .expect_err("subscribe without a loaded model");

    for error in [query_error, chat_error, embed_error, subscribe_error] {
        assert!(matches!(
            error,
            ModelError::ModelNotFound(message) if message == "no model is loaded"
        ));
    }
}
