//! Tests the `lifecycle::browser` module in `sipp`.
//!
//! Covers browser registry, pairing, remote acquisition, cleanup, and
//! observability with pure fixtures instead of native runtime loading.

use super::*;
use crate::lifecycle::AssetRole;
use crate::runtime::config::GpuLayerConfig;

fn inspection(
    role: AssetRole,
    vision_capable: bool,
    compatible: &[&str],
    provided: Option<&str>,
) -> AssetInspection {
    let trained_context_size = (role == AssetRole::Model).then_some(8192);
    AssetInspection {
        version: AssetInspection::VERSION,
        role,
        architecture: Some("test".to_string()),
        trained_context_size,
        vision_capable,
        audio_capable: false,
        audio_generation_capable: false,
        compatible_vision_projector_types: compatible
            .iter()
            .map(|value| value.to_string())
            .collect(),
        compatible_audio_projector_types: Vec::new(),
        compatible_audio_generation_projector_types: Vec::new(),
        provided_vision_projector_type: provided.map(str::to_string),
        provided_audio_projector_type: None,
        provided_audio_generation_projector_type: None,
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

fn service_with_installed_text_model(
    trained_context_size: Option<u32>,
) -> (BrowserLifecycleService, String) {
    let mut model_inspection = inspection(AssetRole::Model, false, &[], None);
    model_inspection.trained_context_size = trained_context_size;
    let model = asset("asset-model", ModelAssetKind::Model, model_inspection);
    let mut service =
        BrowserLifecycleService::create(BrowserCreateConfig { manifest: None }).expect("service");
    let installed = service
        .install(BrowserInstallSource {
            assets: vec![model.clone()],
            classified: vec![classified(&model)],
        })
        .expect("install");
    (service, installed.model.id)
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

    let installed = service
        .install(BrowserInstallSource {
            assets: vec![model.clone()],
            classified: vec![classified(&model)],
        })
        .expect("install");
    let prepared = service
        .prepare_load(
            BrowserLoadSource {
                model_id: installed.model.id,
            },
            load_options(
                json!({ "context": { "n_ctx": 1024 } }),
                BrowserObservabilityMode::Runtime,
            ),
        )
        .expect("prepare");

    assert_eq!(prepared.assets.len(), 1);
    assert!(!prepared.model.asset_fingerprint.is_empty());
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
    assert_eq!(prepared.runtime_config["context"]["n_ctx"], json!(1024));

    let committed = service
        .commit_load(BrowserCommitLoadRequest {
            load_id: prepared.load_id,
            model_id: prepared.model.id.clone(),
            runtime_fingerprint: prepared.runtime_fingerprint,
            runtime: None,
            profile: None,
        })
        .expect("commit");

    assert_eq!(committed.model.id, prepared.model.id);
    assert!(committed.manifest.models[&committed.model.id]
        .last_loaded_at
        .is_some());
}

#[test]
fn browser_cpu_context_uses_the_smaller_trained_capacity() {
    let (mut service, model_id) = service_with_installed_text_model(Some(2048));

    let prepared = service
        .prepare_load(
            BrowserLoadSource { model_id },
            load_options(json!({}), BrowserObservabilityMode::Off),
        )
        .expect("prepare");

    assert_eq!(prepared.runtime_config["context"]["n_ctx"], json!(2048));
}

#[test]
fn browser_cpu_context_caps_large_trained_capacity() {
    let (mut service, model_id) = service_with_installed_text_model(Some(131_072));

    let prepared = service
        .prepare_load(
            BrowserLoadSource { model_id },
            load_options(json!({}), BrowserObservabilityMode::Off),
        )
        .expect("prepare");

    assert_eq!(prepared.runtime_config["context"]["n_ctx"], json!(4096));
}

#[test]
fn browser_explicit_cpu_context_does_not_require_trained_metadata() {
    let (mut service, model_id) = service_with_installed_text_model(None);

    let prepared = service
        .prepare_load(
            BrowserLoadSource { model_id },
            load_options(
                json!({ "context": { "n_ctx": 1024 } }),
                BrowserObservabilityMode::Off,
            ),
        )
        .expect("prepare");

    assert_eq!(prepared.runtime_config["context"]["n_ctx"], json!(1024));
}

#[test]
fn browser_webgpu_context_does_not_require_trained_metadata() {
    let (mut service, model_id) = service_with_installed_text_model(None);

    let prepared = service
        .prepare_load(
            BrowserLoadSource { model_id },
            load_options_with_backend(
                BrowserBackendPreference::WebGpu,
                json!({}),
                BrowserObservabilityMode::Off,
            ),
        )
        .expect("prepare");

    assert!(prepared.runtime_config["context"]["n_ctx"].is_null());
}

#[test]
fn browser_omitted_cpu_context_uses_the_ceiling_without_trained_metadata() {
    let (mut service, model_id) = service_with_installed_text_model(None);

    let prepared = service
        .prepare_load(
            BrowserLoadSource { model_id },
            load_options(json!({}), BrowserObservabilityMode::Off),
        )
        .expect("prepare");

    assert_eq!(prepared.runtime_config["context"]["n_ctx"], json!(4096));
}

#[test]
fn trained_context_resolution_returns_none_when_metadata_is_unavailable() {
    let (service, model_id) = service_with_installed_text_model(None);
    let entry = service.manifest.models[&model_id].clone();

    assert_eq!(
        service
            .trained_context_size_for_entry(&entry)
            .expect("trained context resolution"),
        None
    );
}

#[test]
fn trained_context_resolution_allows_metadata_less_continuation_shards() {
    let (mut service, model_id) = service_with_installed_text_model(Some(2048));
    let continuation = asset(
        "asset-continuation",
        ModelAssetKind::Model,
        AssetInspection::unknown(),
    );
    service
        .manifest
        .assets
        .insert(continuation.id.clone(), continuation);
    let mut entry = service.manifest.models[&model_id].clone();
    entry.model_asset_ids.push("asset-continuation".to_string());

    assert_eq!(
        service
            .trained_context_size_for_entry(&entry)
            .expect("trained context"),
        Some(2048)
    );
}

#[test]
fn trained_context_resolution_rejects_conflicting_asset_metadata() {
    let (mut service, model_id) = service_with_installed_text_model(Some(2048));
    let mut conflicting_inspection = inspection(AssetRole::Model, false, &[], None);
    conflicting_inspection.trained_context_size = Some(4096);
    let conflicting = asset(
        "asset-conflicting",
        ModelAssetKind::Model,
        conflicting_inspection,
    );
    service
        .manifest
        .assets
        .insert(conflicting.id.clone(), conflicting);
    let mut entry = service.manifest.models[&model_id].clone();
    entry.model_asset_ids.push("asset-conflicting".to_string());

    let error = service
        .trained_context_size_for_entry(&entry)
        .expect_err("conflicting trained context");
    assert!(matches!(error, ModelError::InvalidModelSource(message)
    if message == format!(
        "model '{model_id}' has inconsistent trained context metadata across its assets"
    )));
}

#[test]
fn remove_enforces_the_in_use_rule_from_the_caller_supplied_active_model() {
    let model = asset(
        "asset-model",
        ModelAssetKind::Model,
        inspection(AssetRole::Model, false, &[], None),
    );
    let mut service =
        BrowserLifecycleService::create(BrowserCreateConfig { manifest: None }).expect("service");
    let installed = service
        .install(BrowserInstallSource {
            assets: vec![model.clone()],
            classified: vec![classified(&model)],
        })
        .expect("install");

    let error = service
        .remove(BrowserRemoveRequest {
            model_id: installed.model.id.clone(),
            active_model_id: Some(installed.model.id.clone()),
        })
        .expect_err("active model removal");

    assert!(matches!(error, ModelError::ModelInUse(id) if id == installed.model.id));
    assert_eq!(service.list().len(), 1);
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
fn incompatible_projector_failure_restores_previous_entry() {
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
        .install(BrowserInstallSource {
            assets: vec![base.clone(), first_projector.clone()],
            classified: vec![classified(&base), classified(&first_projector)],
        })
        .expect("first install");
    assert_eq!(first.model.status, ModelStatus::Ready);

    let error = service
        .install(BrowserInstallSource {
            assets: vec![bad_projector.clone()],
            classified: vec![classified(&base), classified(&bad_projector)],
        })
        .expect_err("mismatched projector");

    assert!(matches!(error, ModelError::InvalidModelPairing(_)));
    let entry = service.manifest.models.get(&first.model.id).expect("entry");
    assert_eq!(entry.projector_asset_id.as_deref(), Some("asset-mmproj"));
}

#[test]
fn browser_error_response_preserves_unsupported_operation_code() {
    let response: BrowserLifecycleEnvelope<()> = error_response(ModelError::UnsupportedOperation {
        operation: "chat",
        reason: "model has no chat template".to_string(),
    });

    let error = response.error.expect("error");
    assert_eq!(error.code, "UNSUPPORTED_OPERATION");
    assert_eq!(
        error.message,
        "unsupported operation chat: model has no chat template"
    );
}

#[test]
fn browser_remote_commands_use_shared_503_retry_policy_without_cache_fallback() {
    let mut service =
        BrowserLifecycleService::create(BrowserCreateConfig { manifest: None }).expect("service");
    let mut response = service
        .remote_command(BrowserRemoteCommand::Begin {
            urls: vec!["https://example.test/model.gguf".to_string()],
        })
        .expect("begin");

    for attempt in 1..=4 {
        let BrowserRemoteCommandResponse::Action { action } = response else {
            panic!("expected metadata action");
        };
        assert_eq!(action["kind"], "fetch_metadata");
        let acquisition_id = action["acquisitionId"]
            .as_str()
            .expect("acquisition id")
            .to_string();
        let result = service.remote_command(BrowserRemoteCommand::Advance {
            event: json!({
                "kind": "operation_failed",
                "acquisitionId": acquisition_id,
                "memberId": 0,
                "attempt": attempt,
                "failure": {
                    "phase": "metadata",
                    "kind": "http",
                    "status": 503,
                    "reason": "HTTP 503"
                },
                "createdAssetIds": []
            }),
            assets: Vec::new(),
            classified: Vec::new(),
        });
        if attempt == 4 {
            let BrowserRemoteCommandResponse::Failed { error } =
                result.expect("terminal remote failure")
            else {
                panic!("expected failed response");
            };
            assert_eq!(error.code, "REMOTE_METADATA_UNAVAILABLE");
            assert_eq!(error.status, Some(503));
            break;
        }
        let BrowserRemoteCommandResponse::Action { action } = result.expect("wait") else {
            panic!("expected wait action");
        };
        assert_eq!(action["kind"], "wait");
        response = service
            .remote_command(BrowserRemoteCommand::Advance {
                event: json!({
                    "kind": "wait_completed",
                    "acquisitionId": action["acquisitionId"],
                    "memberId": 0,
                    "attempt": attempt
                }),
                assets: Vec::new(),
                classified: Vec::new(),
            })
            .expect("next metadata action");
    }
}

#[test]
fn browser_remote_receipt_failure_cleans_the_created_asset() {
    let mut service =
        BrowserLifecycleService::create(BrowserCreateConfig { manifest: None }).expect("service");
    let BrowserRemoteCommandResponse::Action { action } = service
        .remote_command(BrowserRemoteCommand::Begin {
            urls: vec!["https://example.test/model.gguf".to_string()],
        })
        .expect("begin")
    else {
        panic!("expected metadata action");
    };
    let acquisition_id = action["acquisitionId"]
        .as_str()
        .expect("acquisition id")
        .to_string();
    let BrowserRemoteCommandResponse::Action { action } = service
        .remote_command(BrowserRemoteCommand::Advance {
            event: json!({
                "kind": "metadata_succeeded",
                "acquisitionId": acquisition_id,
                "memberId": 0,
                "attempt": 1,
                "headers": {
                    "contentLength": 8,
                    "etag": "current"
                }
            }),
            assets: Vec::new(),
            classified: Vec::new(),
        })
        .expect("download action")
    else {
        panic!("expected download action");
    };
    let mut record = asset(
        "asset-new",
        ModelAssetKind::Model,
        inspection(AssetRole::Model, false, &[], None),
    );
    record.name = "model.gguf".to_string();
    record.bytes = 8;
    record.source_url = Some("https://example.test/model.gguf".to_string());
    record.source_etag = Some("current".to_string());
    record.source_bytes = Some(8);
    let mut wrong_classification = classified(&record);
    wrong_classification.name = "wrong.gguf".to_string();

    let BrowserRemoteCommandResponse::Action { action } = service
        .remote_command(BrowserRemoteCommand::Advance {
            event: json!({
                "kind": "download_succeeded",
                "acquisitionId": action["acquisitionId"],
                "memberId": 0,
                "attempt": 1,
                "assetIds": ["asset-new"],
                "createdAssetIds": ["asset-new"]
            }),
            assets: vec![record],
            classified: vec![wrong_classification],
        })
        .expect("cleanup action")
    else {
        panic!("expected cleanup action");
    };
    assert_eq!(action["kind"], "cleanup");
    assert_eq!(action["assetIds"], json!(["asset-new"]));

    let result = service
        .remote_command(BrowserRemoteCommand::Advance {
            event: json!({
                "kind": "cleanup_succeeded",
                "acquisitionId": action["acquisitionId"],
                "memberId": 0,
                "attempt": 1
            }),
            assets: Vec::new(),
            classified: Vec::new(),
        })
        .expect("failed response");
    let BrowserRemoteCommandResponse::Failed { error } = result else {
        panic!("expected failed response");
    };
    assert_eq!(error.code, "INVALID_MODEL_SOURCE");
}

#[test]
fn browser_remote_begin_does_not_replace_an_active_acquisition() {
    let mut service =
        BrowserLifecycleService::create(BrowserCreateConfig { manifest: None }).expect("service");
    service
        .remote_command(BrowserRemoteCommand::Begin {
            urls: vec!["https://example.test/first.gguf".to_string()],
        })
        .expect("first begin");

    let error = service
        .remote_command(BrowserRemoteCommand::Begin {
            urls: vec!["https://example.test/second.gguf".to_string()],
        })
        .expect_err("second begin");

    assert!(matches!(error, ModelError::InvalidModelSource(_)));
}

#[test]
fn browser_remote_error_envelope_preserves_http_retry_metadata() {
    let response: BrowserLifecycleEnvelope<()> =
        error_response(ModelError::RemoteMetadataUnavailable {
            url: "https://example.test/model.gguf".to_string(),
            status: Some(503),
            retry_after_ms: Some(2_000),
            reason: "HTTP 503".to_string(),
        });

    let error = response.error.expect("error");
    assert_eq!(error.code, "REMOTE_METADATA_UNAVAILABLE");
    assert_eq!(error.status, Some(503));
    assert_eq!(error.retry_after_ms, Some(2_000));
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
fn rejects_previous_registry_manifest_versions() {
    let previous_version = REGISTRY_MANIFEST_VERSION - 1;
    let manifest = BrowserRegistryManifest {
        version: previous_version,
        ..BrowserRegistryManifest::default()
    };
    let error = BrowserLifecycleService::create(BrowserCreateConfig {
        manifest: Some(manifest),
    })
    .expect_err("previous manifest version");

    assert!(matches!(
        error,
        ModelError::StorageCorrupt(message)
            if message == format!(
                "expected browser registry manifest version {REGISTRY_MANIFEST_VERSION}, got {previous_version}"
            )
    ));
}

#[test]
fn rejects_previous_asset_inspection_versions() {
    let mut old_inspection = inspection(AssetRole::Model, false, &[], None);
    let previous_version = AssetInspection::VERSION - 1;
    old_inspection.version = previous_version;
    let old_asset = asset("asset", ModelAssetKind::Model, old_inspection);
    let mut manifest = BrowserRegistryManifest::default();
    manifest.assets.insert(old_asset.id.clone(), old_asset);

    let error = BrowserLifecycleService::create(BrowserCreateConfig {
        manifest: Some(manifest),
    })
    .expect_err("previous inspection version");

    assert!(matches!(
        error,
        ModelError::StorageCorrupt(message)
            if message == format!(
                "expected asset inspection version {}, got {previous_version}",
                AssetInspection::VERSION
            )
    ));
}

#[test]
fn unix_epoch_formats_as_iso_string() {
    assert_eq!(iso_from_unix_ms(0), "1970-01-01T00:00:00.000Z");
    assert_eq!(iso_from_unix_ms(1_234), "1970-01-01T00:00:01.234Z");
}
