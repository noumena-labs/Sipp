//! Tests the `lifecycle::browser` module in `sipp`.
//!
//! Covers lifecycle registry, storage, browser, service, and pairing behavior with temporary storage and pure fixtures instead of native runtime loading.

use super::*;
use crate::lifecycle::AssetRole;
use crate::runtime::config::GpuLayerConfig;

fn inspection(
    role: AssetRole,
    vision_capable: bool,
    compatible: &[&str],
    provided: Option<&str>,
) -> AssetInspection {
    AssetInspection {
        version: 1,
        role,
        architecture: Some("test".to_string()),
        vision_capable,
        compatible_vision_projector_types: compatible
            .iter()
            .map(|value| value.to_string())
            .collect(),
        provided_vision_projector_type: provided.map(str::to_string),
    }
}

fn asset(id: &str, kind: ModelAssetKind, inspection: AssetInspection) -> BrowserAssetRecord {
    BrowserAssetRecord {
        id: id.to_string(),
        kind,
        name: format!("{id}.gguf"),
        bytes: 4,
        storage_path: id.to_string(),
        source_url: None,
        source_etag: None,
        source_last_modified: None,
        source_bytes: None,
        source_part_index: None,
        source_part_count: None,
        source_file_name: None,
        source_file_last_modified: None,
        ref_count: 0,
        created_at: "1970-01-01T00:00:00.000Z".to_string(),
        inspection: Some(inspection),
    }
}

fn classified(record: &BrowserAssetRecord) -> ClassifiedAsset {
    classified_asset(
        record.id.clone(),
        record.name.clone(),
        record.inspection.clone(),
    )
}

fn load_options(runtime: Value, observability: BrowserObservabilityMode) -> BrowserLoadOptions {
    load_options_with_backend(BrowserBackendPreference::Cpu, runtime, observability)
}

fn load_options_with_backend(
    backend: BrowserBackendPreference,
    runtime: Value,
    observability: BrowserObservabilityMode,
) -> BrowserLoadOptions {
    BrowserLoadOptions {
        backend,
        runtime,
        observability,
    }
}

fn backend_capabilities(compiled: &[&str], available: &[&str]) -> BackendCapabilities {
    BackendCapabilities {
        compiled: compiled.iter().map(|value| value.to_string()).collect(),
        available: available.iter().map(|value| value.to_string()).collect(),
        gpu_offload_supported: compiled.iter().any(|value| *value != "cpu"),
    }
}

#[test]
fn prepares_and_commits_text_load() {
    let model = asset(
        "asset-model",
        ModelAssetKind::Model,
        inspection(AssetRole::Model, false, &[], None),
    );
    let mut service =
        BrowserLifecycleService::create(BrowserCreateConfig { manifest: None }).expect("service");

    let prepared = service
        .prepare_load(
            BrowserLoadSource::Assets {
                assets: vec![model.clone()],
                classified: vec![classified(&model)],
                explicit_projector_asset_id: None,
                classified_projectors: Vec::new(),
            },
            load_options(
                json!({ "context": { "n_ctx": 1024 } }),
                BrowserObservabilityMode::Runtime,
            ),
        )
        .expect("prepare");

    assert!(prepared.load_required);
    assert_eq!(prepared.assets.len(), 1);
    assert_eq!(prepared.model.status, ModelStatus::Ready);
    assert_eq!(prepared.manifest.assets["asset-model"].ref_count, 1);
    assert_eq!(
        prepared.runtime_config["placement"]["gpu_layers"],
        json!({ "count": 0 })
    );
    assert_eq!(
        prepared.runtime_config["placement"]["split_mode"],
        json!("layer")
    );
    assert_eq!(
        prepared.runtime_config["observability"]["runtime_metrics"],
        json!(true)
    );
    assert_eq!(prepared.runtime_config["context"]["warmup"], json!(true));

    let committed = service
        .commit_load(BrowserCommitLoadRequest {
            load_id: prepared.load_id,
            model_id: prepared.model.id,
            runtime_fingerprint: prepared.runtime_fingerprint,
            chat_template: Some("template".to_string()),
            bos_text: "<s>".to_string(),
            eos_text: "</s>".to_string(),
            media_marker: None,
            runtime: None,
            profile: None,
        })
        .expect("commit");

    assert!(committed.model.loaded);
    assert_eq!(committed.model.chat_template.as_deref(), Some("template"));
}

#[test]
fn browser_runtime_preserves_explicit_warmup() {
    let plan = browser_backend_plan(&load_options(
        json!({ "context": { "warmup": true } }),
        BrowserObservabilityMode::Off,
    ))
    .expect("plan");

    assert!(plan.config.context.warmup);
}

#[test]
fn browser_auto_without_capability_probe_defaults_cpu() {
    let plan = browser_backend_plan(&load_options_with_backend(
        BrowserBackendPreference::Auto,
        json!({}),
        BrowserObservabilityMode::Off,
    ))
    .expect("auto plan");

    assert_eq!(plan.selection.requested, BackendPreference::Auto);
    assert_eq!(plan.selection.selected, "cpu");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Count(0));
}

#[test]
fn browser_webgpu_without_capability_probe_uses_full_offload() {
    let plan = browser_backend_plan(&load_options_with_backend(
        BrowserBackendPreference::WebGpu,
        json!({}),
        BrowserObservabilityMode::Off,
    ))
    .expect("webgpu plan");

    assert_eq!(plan.selection.requested, BackendPreference::WebGpu);
    assert_eq!(plan.selection.selected, "webgpu");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Auto);
    assert!(plan.selection.gpu_offload_expected);
}

#[test]
fn browser_auto_selects_webgpu_when_capable() {
    let plan = browser_backend_plan_with_capabilities(
        &load_options_with_backend(
            BrowserBackendPreference::Auto,
            json!({}),
            BrowserObservabilityMode::Off,
        ),
        Some(&backend_capabilities(&["webgpu"], &["cpu", "webgpu"])),
    )
    .expect("webgpu plan");

    assert_eq!(plan.selection.requested, BackendPreference::Auto);
    assert_eq!(plan.selection.selected, "webgpu");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Auto);
    assert!(plan.selection.gpu_offload_expected);
}

#[test]
fn browser_cpu_forces_cpu_when_webgpu_is_capable() {
    let plan = browser_backend_plan_with_capabilities(
        &load_options_with_backend(
            BrowserBackendPreference::Cpu,
            json!({}),
            BrowserObservabilityMode::Off,
        ),
        Some(&backend_capabilities(&["webgpu"], &["cpu", "webgpu"])),
    )
    .expect("cpu plan");

    assert_eq!(plan.selection.selected, "cpu");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Count(0));
    assert!(!plan.selection.gpu_offload_expected);
}

#[test]
fn browser_webgpu_requires_available_backend() {
    let error = browser_backend_plan_with_capabilities(
        &load_options_with_backend(
            BrowserBackendPreference::WebGpu,
            json!({}),
            BrowserObservabilityMode::Off,
        ),
        Some(&backend_capabilities(&[], &["cpu"])),
    )
    .expect_err("missing webgpu");

    assert!(matches!(error, ModelError::InvalidModelSource(_)));
}

#[test]
fn explicit_projector_failure_restores_previous_entry() {
    let base = asset(
        "asset-base",
        ModelAssetKind::Model,
        inspection(AssetRole::Model, true, &["lfm2"], None),
    );
    let first_projector = asset(
        "asset-mmproj",
        ModelAssetKind::Projector,
        inspection(AssetRole::Projector, false, &[], Some("lfm2")),
    );
    let bad_projector = asset(
        "asset-bad",
        ModelAssetKind::Projector,
        inspection(AssetRole::Projector, false, &[], Some("other")),
    );
    let mut service =
        BrowserLifecycleService::create(BrowserCreateConfig { manifest: None }).expect("service");

    let first = service
        .prepare_load(
            BrowserLoadSource::Assets {
                assets: vec![base.clone(), first_projector.clone()],
                classified: vec![classified(&base), classified(&first_projector)],
                explicit_projector_asset_id: Some(first_projector.id.clone()),
                classified_projectors: Vec::new(),
            },
            load_options(json!({}), BrowserObservabilityMode::Off),
        )
        .expect("first prepare");
    assert_eq!(first.model.status, ModelStatus::Ready);

    let error = service
        .prepare_load(
            BrowserLoadSource::Assets {
                assets: vec![bad_projector.clone()],
                classified: vec![classified(&base), classified(&bad_projector)],
                explicit_projector_asset_id: Some(bad_projector.id),
                classified_projectors: Vec::new(),
            },
            load_options(json!({}), BrowserObservabilityMode::Off),
        )
        .expect_err("mismatched projector");

    assert!(matches!(error, ModelError::InvalidModelPairing(_)));
    let entry = service.manifest.models.get(&first.model.id).expect("entry");
    assert_eq!(entry.projector_asset_id.as_deref(), Some("asset-mmproj"));
}

#[test]
fn abort_load_records_failed_query_observation() {
    let mut service =
        BrowserLifecycleService::create(BrowserCreateConfig { manifest: None }).expect("service");

    let snapshot = service.abort_load(Some("load failed".to_string()));

    assert_eq!(snapshot.state, BrowserLifecycleState::Error);
    let query = snapshot.query.expect("query observation");
    assert_eq!(query.status, QUERY_STATUS_FAILED);
    assert_eq!(query.error_code.as_deref(), Some(CODE_QUERY_FAILED));
    assert_eq!(query.error_message.as_deref(), Some("load failed"));
}

#[test]
fn browser_error_response_preserves_unsupported_operation_code() {
    let response: BrowserLifecycleEnvelope<()> = error_response(ModelError::UnsupportedOperation {
        operation: "chat",
        reason: "model has no chat template".to_string(),
    });

    let error = response.error.expect("error");
    assert_eq!(error.code, CODE_UNSUPPORTED_OPERATION);
    assert_eq!(
        error.message,
        "unsupported operation chat: model has no chat template"
    );
}

#[test]
fn success_response_and_response_json_preserve_envelope_shape() {
    let response = success_response(json!({ "value": 7 }));

    assert!(response.ok);
    assert!(response.error.is_none());

    let rendered = response_json(response);
    let value: Value = serde_json::from_str(&rendered).expect("json envelope");
    assert_eq!(value["ok"], json!(true));
    assert_eq!(value["value"]["value"], json!(7));
}

#[test]
fn validate_manifest_rejects_asset_key_mismatch() {
    let record = asset(
        "asset-a",
        ModelAssetKind::Model,
        inspection(AssetRole::Model, false, &[], None),
    );
    let mut manifest = BrowserRegistryManifest::default();
    manifest.assets.insert("wrong-key".to_string(), record);

    let error = validate_manifest(&manifest).expect_err("mismatched asset key");

    assert!(matches!(
        error,
        ModelError::StorageCorrupt(message) if message.contains("does not match")
    ));
}

#[test]
fn snapshot_patch_updates_supplied_fields_and_preserves_others() {
    let service =
        BrowserLifecycleService::create(BrowserCreateConfig { manifest: None }).expect("service");
    let original = service.snapshot.clone();

    let patched = apply_snapshot_patch(
        original.clone(),
        SnapshotPatch {
            mode: Some(BrowserObservabilityMode::Profile),
            state: Some(BrowserLifecycleState::Ready),
            runtime: Some(Some(json!({ "decodeMs": 1.0 }))),
            ..SnapshotPatch::default()
        },
    );

    assert_eq!(patched.mode, BrowserObservabilityMode::Profile);
    assert_eq!(patched.state, BrowserLifecycleState::Ready);
    assert_eq!(patched.runtime, Some(json!({ "decodeMs": 1.0 })));
    assert_eq!(patched.model, original.model);
    assert_eq!(patched.query, original.query);
}

#[test]
fn migrates_existing_v3_split_remote_shape_without_losing_order() {
    let raw = r#"{
      "version": 3,
      "projectorIndexRevision": 0,
      "assets": {
        "asset-a": {
          "id": "asset-a",
          "kind": "shard",
          "name": "part-1.gguf",
          "hash": "a",
          "bytes": 4,
          "storagePath": "part-1.gguf",
          "sourceUrl": "https://example.test/model.gguf",
          "sourceEtag": "\"abc\"",
          "sourceLastModified": "Thu, 01 Jan 1970 00:00:00 GMT",
          "sourceBytes": 8,
          "sourcePartIndex": 0,
          "sourcePartCount": 2,
          "refCount": 0,
          "createdAt": "1970-01-01T00:00:00.000Z"
        }
      },
      "models": {}
    }"#;
    let manifest: BrowserRegistryManifest = serde_json::from_str(raw).expect("manifest");
    let service = BrowserLifecycleService::create(BrowserCreateConfig {
        manifest: Some(manifest),
    })
    .expect("service");
    let asset = &service.manifest.assets["asset-a"];
    assert_eq!(asset.source_part_index, Some(0));
    assert_eq!(asset.source_part_count, Some(2));
    assert_eq!(
        asset.source_url.as_deref(),
        Some("https://example.test/model.gguf")
    );
}

#[test]
fn unix_epoch_formats_as_iso_string() {
    assert_eq!(iso_from_unix_ms(0), "1970-01-01T00:00:00.000Z");
    assert_eq!(iso_from_unix_ms(1_234), "1970-01-01T00:00:01.234Z");
}
