//! Browser lifecycle catalog.
//!
//! This module owns the browser-facing model registry contract without taking a
//! dependency on OPFS, `File`, `fetch`, or WORKERFS. The browser host installs
//! assets and mounts files; Rust owns the persisted lifecycle decisions.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::collection::{
    remove_matching_values, sorted_ref_deltas, sorted_unique_strings, sorted_values,
};
use crate::lifecycle::acquisition::{
    canonical_remote_url, RemoteAcquisition, RemoteAcquisitionEvent, RemoteAcquisitionIds,
    RemoteAcquisitionProgress, RemoteAcquisitionRequest, RemoteAction, RemoteCacheCandidate,
    RemoteMetadata,
};
use crate::lifecycle::util::{
    asset_refcount_mismatch, asset_summary, bump_projector_index_revision as bump_revision,
    classified_asset, decrement_asset_refcount, empty_asset_id, increment_asset_refcount,
    increment_expected_asset_refcount, invalid_asset_field, invalid_source, manifest_key_mismatch,
    missing_model_asset, model_missing_asset, model_not_found, sha256_hex, sorted_model_asset_ids,
    validate_asset_inspection_version, validate_registry_manifest_version, AssetSummary,
};
use crate::lifecycle::{
    AssetInspection, BackendCapabilities, BackendPlan, BackendPolicy, BackendPreference,
    BackendSelection, ClassifiedAsset, ModelAssetKind, ModelError, ModelLoadOptions, ModelModality,
    ModelPairingReason, ModelPairingState as CoreModelPairingState, ModelSourceKind, ModelStatus,
    PairingPlan, PairingResolver, StatsMode, REGISTRY_MANIFEST_VERSION,
};
use crate::runtime::config::NativeRuntimeConfig;
use crate::runtime::numeric::{
    MILLIS_PER_SECOND, SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE,
};

const BROWSER_CPU_CONTEXT_CEILING: u32 = 4096;
const NO_PENDING_BROWSER_MODEL_LOAD: &str = "no pending browser model load";
const BROWSER_MODEL_LOAD_ALREADY_PENDING: &str = "a browser model load is already pending";
const BROWSER_LOAD_COMMIT_MISMATCH: &str =
    "browser model load commit does not match the pending load";
const BROWSER_REGISTRY_MANIFEST_LABEL: &str = "browser registry manifest";
const BROWSER_MODEL_ID_HASH_CHARS: usize = 24;
const LIFECYCLE_SERIALIZATION_FALLBACK: &str =
    "{\"ok\":false,\"error\":{\"code\":\"STORAGE_CORRUPT\",\"message\":\"failed to serialize lifecycle response\"}}";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserRegistryManifest {
    pub version: u32,
    #[serde(default)]
    pub projector_index_revision: u64,
    #[serde(default)]
    pub assets: BTreeMap<String, BrowserAssetRecord>,
    #[serde(default)]
    pub models: BTreeMap<String, BrowserModelEntry>,
}

impl Default for BrowserRegistryManifest {
    fn default() -> Self {
        Self {
            version: REGISTRY_MANIFEST_VERSION,
            projector_index_revision: 0,
            assets: BTreeMap::new(),
            models: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserAssetRecord {
    pub id: String,
    pub kind: ModelAssetKind,
    pub name: String,
    pub bytes: u64,
    pub storage_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_etag: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_last_modified: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_bytes: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_part_index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_part_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_file_last_modified: Option<u64>,
    #[serde(default)]
    pub ref_count: u32,
    pub created_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inspection: Option<AssetInspection>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserModelPairing {
    pub state: CoreModelPairingState,
    pub checked_projector_index_revision: u64,
    pub compatible_vision_projector_types: Vec<String>,
    /// Audio-input projector types accepted by the model.
    pub compatible_audio_projector_types: Vec<String>,
    /// Audio-generation projector types accepted by the model.
    pub compatible_audio_generation_projector_types: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason_code: Option<ModelPairingReason>,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserModelEntry {
    pub id: String,
    pub name: String,
    pub modality: ModelModality,
    pub status: ModelStatus,
    #[serde(default)]
    pub model_asset_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector_asset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pairing: Option<BrowserModelPairing>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_fingerprint: Option<String>,
    pub created_at: String,
    pub updated_at: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_loaded_at: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserModelInfo {
    pub id: String,
    pub name: String,
    pub modality: ModelModality,
    pub status: ModelStatus,
    pub source: ModelSourceKind,
    pub bytes: u64,
    pub asset_fingerprint: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrowserLifecycleState {
    Idle,
    Loading,
    Ready,
    Querying,
    Error,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BrowserObservabilityMode {
    #[default]
    Off,
    Runtime,
    Profile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BrowserBackendPreference {
    #[default]
    Auto,
    Cpu,
    #[serde(rename = "webgpu")]
    WebGpu,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserQueryObservation {
    pub context_key: Option<String>,
    pub status: String,
    pub wall_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserObservabilitySnapshot {
    pub mode: BrowserObservabilityMode,
    pub state: BrowserLifecycleState,
    pub updated_at: String,
    pub model: Option<BrowserModelInfo>,
    pub query: Option<BrowserQueryObservation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BrowserObservabilityEventType {
    LoadStart,
    LoadComplete,
    QueryStart,
    QueryComplete,
    Error,
    Close,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserObservabilityEvent {
    #[serde(rename = "type")]
    pub event_type: BrowserObservabilityEventType,
    pub snapshot: BrowserObservabilitySnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserCreateConfig {
    #[serde(default)]
    pub manifest: Option<BrowserRegistryManifest>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct BrowserLoadOptions {
    #[serde(default)]
    pub backend: BrowserBackendPreference,
    #[serde(default)]
    pub runtime: Value,
    #[serde(default)]
    pub observability: BrowserObservabilityMode,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserLoadSource {
    pub model_id: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserInstallSource {
    pub assets: Vec<BrowserAssetRecord>,
    pub classified: Vec<ClassifiedAsset>,
}

/// Browser host command for Rust-owned remote acquisition.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(
    tag = "command",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum BrowserRemoteCommand {
    Begin {
        urls: Vec<String>,
    },
    Advance {
        event: Value,
        #[serde(default)]
        assets: Vec<BrowserAssetRecord>,
        #[serde(default)]
        classified: Vec<ClassifiedAsset>,
    },
    Cancel {
        acquisition_id: String,
    },
}

/// Rust acquisition response consumed by the browser I/O executor.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(
    tag = "kind",
    rename_all = "snake_case",
    rename_all_fields = "camelCase"
)]
pub enum BrowserRemoteCommandResponse {
    Action {
        action: Value,
    },
    Installed {
        installed: Box<BrowserInstallResponse>,
    },
    Cancelled {
        snapshot: Box<BrowserObservabilitySnapshot>,
    },
    Failed {
        error: Box<BrowserLifecycleError>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserPlannedAsset {
    pub asset_id: String,
    pub kind: ModelAssetKind,
    pub storage_path: String,
    pub mount_name: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserPrepareLoadResponse {
    pub load_id: String,
    pub model: BrowserModelInfo,
    pub runtime_fingerprint: String,
    pub runtime_config: Value,
    #[serde(default)]
    pub assets: Vec<BrowserPlannedAsset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector: Option<BrowserPlannedAsset>,
    pub manifest: BrowserRegistryManifest,
    pub snapshot: BrowserObservabilitySnapshot,
    #[serde(default)]
    pub events: Vec<BrowserObservabilityEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserInstallResponse {
    pub model: BrowserModelInfo,
    pub manifest: BrowserRegistryManifest,
    pub snapshot: BrowserObservabilitySnapshot,
    #[serde(default)]
    pub events: Vec<BrowserObservabilityEvent>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserCommitLoadRequest {
    pub load_id: String,
    pub model_id: String,
    pub runtime_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserCommitLoadResponse {
    pub model: BrowserModelInfo,
    pub manifest: BrowserRegistryManifest,
    pub snapshot: BrowserObservabilitySnapshot,
    #[serde(default)]
    pub events: Vec<BrowserObservabilityEvent>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserRemoveResponse {
    pub removed: BrowserModelEntry,
    #[serde(default)]
    pub orphaned_assets: Vec<BrowserAssetRecord>,
    pub manifest: BrowserRegistryManifest,
    pub snapshot: BrowserObservabilitySnapshot,
    #[serde(default)]
    pub events: Vec<BrowserObservabilityEvent>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserRemoveRequest {
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_model_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserLifecycleEnvelope<T>
where
    T: Serialize,
{
    pub ok: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<BrowserLifecycleError>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserLifecycleError {
    pub code: &'static str,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<u64>,
}

#[derive(Debug, Clone)]
struct PendingLoad {
    load_id: String,
    model_id: String,
    runtime_fingerprint: String,
}

#[derive(Debug)]
struct PendingRemoteAcquisition {
    acquisition: RemoteAcquisition,
    assets: BTreeMap<String, BrowserAssetRecord>,
    classified: BTreeMap<String, ClassifiedAsset>,
}

struct BrowserResponseContext {
    manifest: BrowserRegistryManifest,
    snapshot: BrowserObservabilitySnapshot,
    events: Vec<BrowserObservabilityEvent>,
}

#[derive(Debug)]
pub struct BrowserLifecycleService {
    pub manifest: BrowserRegistryManifest,
    pending: Option<PendingLoad>,
    pending_remote: Option<PendingRemoteAcquisition>,
    acquisition_ids: RemoteAcquisitionIds,
    pub snapshot: BrowserObservabilitySnapshot,
    events: VecDeque<BrowserObservabilityEvent>,
}

impl BrowserLifecycleService {
    pub fn create(config: BrowserCreateConfig) -> Result<Self, ModelError> {
        let manifest = config.manifest.unwrap_or_default();
        validate_manifest(&manifest)?;
        let now = now_iso();
        Ok(Self {
            manifest,
            pending: None,
            pending_remote: None,
            acquisition_ids: RemoteAcquisitionIds::default(),
            snapshot: BrowserObservabilitySnapshot {
                mode: BrowserObservabilityMode::Off,
                state: BrowserLifecycleState::Idle,
                updated_at: now,
                model: None,
                query: None,
                runtime: None,
                profile: None,
            },
            events: VecDeque::new(),
        })
    }

    pub fn list(&self) -> Vec<BrowserModelInfo> {
        self.manifest
            .models
            .values()
            .map(|entry| self.model_info_from_entry(entry))
            .collect()
    }

    pub fn drain_events(&mut self) -> Vec<BrowserObservabilityEvent> {
        self.events.drain(..).collect()
    }

    pub fn prepare_load(
        &mut self,
        source: BrowserLoadSource,
        options: BrowserLoadOptions,
    ) -> Result<BrowserPrepareLoadResponse, ModelError> {
        self.pending_remote = None;
        validate_manifest(&self.manifest)?;
        let entry = self
            .manifest
            .models
            .get(&source.model_id)
            .cloned()
            .ok_or_else(|| model_not_found(&source.model_id))?;
        let base_plan = self.derive_base_plan_for_entry(&entry)?;
        let entry = self.resolve_entry_for_loading(entry, &base_plan)?;

        let mut backend_plan = browser_backend_plan(&options)?;
        self.apply_browser_cpu_context_policy(&entry, &mut backend_plan)?;
        let runtime_config = serde_json::to_value(&backend_plan.config)?;
        let runtime_fingerprint = runtime_fingerprint(&runtime_config, &backend_plan.selection);
        let asset_fingerprint = asset_fingerprint(&entry);
        let load_id = browser_load_id(&entry.id, &asset_fingerprint, &runtime_fingerprint);
        let model = self.model_info_from_entry(&entry);
        let (assets, projector) = self.planned_assets_for_entry(&entry)?;
        self.pending = Some(PendingLoad {
            load_id: load_id.clone(),
            model_id: entry.id.clone(),
            runtime_fingerprint: runtime_fingerprint.clone(),
        });
        self.emit(
            BrowserObservabilityEventType::LoadStart,
            SnapshotPatch {
                mode: Some(options.observability),
                state: Some(BrowserLifecycleState::Loading),
                model: None,
                query: Some(None),
                runtime: Some(None),
                profile: Some(None),
            },
        );
        let response = self.response_context();

        Ok(BrowserPrepareLoadResponse {
            load_id,
            model,
            runtime_fingerprint,
            runtime_config,
            assets,
            projector,
            manifest: response.manifest,
            snapshot: response.snapshot,
            events: response.events,
        })
    }

    fn apply_browser_cpu_context_policy(
        &self,
        entry: &BrowserModelEntry,
        plan: &mut BackendPlan,
    ) -> Result<(), ModelError> {
        if plan.selection.selected != BackendPreference::Cpu.as_str()
            || plan.config.context.n_ctx.is_some()
        {
            return Ok(());
        }

        let context_size = self
            .trained_context_size_for_entry(entry)?
            .unwrap_or(BROWSER_CPU_CONTEXT_CEILING)
            .min(BROWSER_CPU_CONTEXT_CEILING);
        plan.config.context.n_ctx = Some(context_size as i32);
        Ok(())
    }

    fn trained_context_size_for_entry(
        &self,
        entry: &BrowserModelEntry,
    ) -> Result<Option<u32>, ModelError> {
        let mut trained_context_size = None;
        for asset_id in &entry.model_asset_ids {
            let asset = self
                .manifest
                .assets
                .get(asset_id)
                .ok_or_else(|| missing_model_asset(asset_id))?;
            let Some(size) = asset
                .inspection
                .as_ref()
                .and_then(|inspection| inspection.trained_context_size)
            else {
                continue;
            };
            if trained_context_size.is_some_and(|current| current != size) {
                return Err(ModelError::InvalidModelSource(format!(
                    "model '{}' has inconsistent trained context metadata across its assets",
                    entry.id
                )));
            }
            trained_context_size = Some(size);
        }

        Ok(trained_context_size)
    }

    pub fn install(
        &mut self,
        source: BrowserInstallSource,
    ) -> Result<BrowserInstallResponse, ModelError> {
        let previous = self.manifest.clone();
        let result = self.install_inner(source);
        if result.is_err() {
            self.manifest = previous;
        }
        result
    }

    fn install_inner(
        &mut self,
        source: BrowserInstallSource,
    ) -> Result<BrowserInstallResponse, ModelError> {
        validate_manifest(&self.manifest)?;
        self.upsert_assets(source.assets, &source.classified)?;
        let plan = PairingResolver::resolve(&source.classified)?;
        let source_projector = plan.projector_asset_id.clone();
        let base_plan = if let Some(projector_id) = source_projector.as_deref() {
            let base_assets: Vec<_> = source
                .classified
                .iter()
                .filter(|asset| asset.asset_id != projector_id)
                .cloned()
                .collect();
            PairingResolver::resolve(&base_assets)?
        } else {
            plan.clone()
        };
        let mut entry = self.upsert_base_model_entry(&base_plan)?;
        if let Some(projector_id) = source_projector {
            entry = self.set_resolved_projector(&entry.id, &projector_id, &plan)?;
        } else {
            entry = self.resolve_entry_for_loading(entry, &base_plan)?;
        }
        validate_manifest(&self.manifest)?;
        let model = self.model_info_from_entry(&entry);
        let response = self.response_context();
        Ok(BrowserInstallResponse {
            model,
            manifest: response.manifest,
            snapshot: response.snapshot,
            events: response.events,
        })
    }

    pub fn remote_command(
        &mut self,
        command: BrowserRemoteCommand,
    ) -> Result<BrowserRemoteCommandResponse, ModelError> {
        match command {
            BrowserRemoteCommand::Begin { urls } => self.begin_remote(urls),
            BrowserRemoteCommand::Advance {
                event,
                assets,
                classified,
            } => self.advance_remote(event, assets, classified),
            BrowserRemoteCommand::Cancel { acquisition_id } => self.cancel_remote(acquisition_id),
        }
    }

    fn begin_remote(
        &mut self,
        urls: Vec<String>,
    ) -> Result<BrowserRemoteCommandResponse, ModelError> {
        if urls.is_empty() {
            return Err(invalid_source("remote model sources must not be empty"));
        }
        if self.pending.is_some() || self.pending_remote.is_some() {
            return Err(invalid_source(BROWSER_MODEL_LOAD_ALREADY_PENDING));
        }
        let acquisition_id = self.acquisition_ids.issue()?;
        let requests = browser_remote_requests(&self.manifest, urls)?;
        let acquisition = RemoteAcquisition::new(acquisition_id, requests)?;
        self.pending_remote = Some(PendingRemoteAcquisition {
            acquisition,
            assets: BTreeMap::new(),
            classified: BTreeMap::new(),
        });
        self.remote_progress()
    }

    fn advance_remote(
        &mut self,
        event: Value,
        assets: Vec<BrowserAssetRecord>,
        classified: Vec<ClassifiedAsset>,
    ) -> Result<BrowserRemoteCommandResponse, ModelError> {
        let event = serde_json::from_value::<RemoteAcquisitionEvent>(event)?;
        let progress = {
            let pending = self
                .pending_remote
                .as_mut()
                .ok_or_else(|| invalid_source("no remote acquisition is pending"))?;
            let action = match pending.acquisition.progress() {
                RemoteAcquisitionProgress::Action(action) => action,
                _ => {
                    return Err(invalid_source(
                        "remote acquisition is not waiting for a host event",
                    ));
                }
            };
            pending.acquisition.validate_event(&event)?;
            let progress = pending.acquisition.advance(event.clone())?;
            match accept_remote_payload(
                pending,
                &self.manifest,
                &action,
                &event,
                assets,
                classified,
            ) {
                Ok(()) => progress,
                Err(error) => pending.acquisition.fail_finalization(error),
            }
        };
        self.resolve_remote_progress(progress)
    }

    fn cancel_remote(
        &mut self,
        acquisition_id: String,
    ) -> Result<BrowserRemoteCommandResponse, ModelError> {
        let progress = {
            let pending = self
                .pending_remote
                .as_mut()
                .ok_or_else(|| invalid_source("no remote acquisition is pending"))?;
            pending
                .acquisition
                .advance(RemoteAcquisitionEvent::Cancelled { acquisition_id })?
        };
        self.resolve_remote_progress(progress)
    }

    fn remote_progress(&mut self) -> Result<BrowserRemoteCommandResponse, ModelError> {
        let progress = self
            .pending_remote
            .as_mut()
            .ok_or_else(|| invalid_source("no remote acquisition is pending"))?
            .acquisition
            .progress();
        self.resolve_remote_progress(progress)
    }

    fn resolve_remote_progress(
        &mut self,
        progress: RemoteAcquisitionProgress,
    ) -> Result<BrowserRemoteCommandResponse, ModelError> {
        match progress {
            RemoteAcquisitionProgress::Action(action) => Ok(BrowserRemoteCommandResponse::Action {
                action: serde_json::to_value(action)?,
            }),
            RemoteAcquisitionProgress::Ready(resolved) => self.finalize_remote(resolved),
            RemoteAcquisitionProgress::Failed(error) => {
                self.pending_remote = None;
                Ok(BrowserRemoteCommandResponse::Failed {
                    error: Box::new(lifecycle_error(error)),
                })
            }
            RemoteAcquisitionProgress::Cancelled => {
                self.pending_remote = None;
                Ok(BrowserRemoteCommandResponse::Cancelled {
                    snapshot: Box::new(self.snapshot.clone()),
                })
            }
        }
    }

    fn finalize_remote(
        &mut self,
        resolved: Vec<crate::lifecycle::acquisition::RemoteResolvedMember>,
    ) -> Result<BrowserRemoteCommandResponse, ModelError> {
        let mut pending = self
            .pending_remote
            .take()
            .ok_or_else(|| invalid_source("no remote acquisition is pending"))?;
        let asset_ids: Vec<_> = resolved
            .iter()
            .flat_map(|member| member.asset_ids.iter().cloned())
            .collect();
        let assets = asset_ids
            .iter()
            .map(|asset_id| {
                pending
                    .assets
                    .get(asset_id)
                    .cloned()
                    .ok_or_else(|| missing_model_asset(asset_id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let classified = asset_ids
            .iter()
            .map(|asset_id| {
                pending
                    .classified
                    .get(asset_id)
                    .cloned()
                    .ok_or_else(|| invalid_source("remote asset classification is missing"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        match self.install(BrowserInstallSource { assets, classified }) {
            Ok(installed) => Ok(BrowserRemoteCommandResponse::Installed {
                installed: Box::new(installed),
            }),
            Err(error) => {
                let progress = pending.acquisition.fail_finalization(error);
                self.pending_remote = Some(pending);
                self.resolve_remote_progress(progress)
            }
        }
    }

    pub fn commit_load(
        &mut self,
        request: BrowserCommitLoadRequest,
    ) -> Result<BrowserCommitLoadResponse, ModelError> {
        let pending = self
            .pending
            .take()
            .ok_or_else(|| invalid_source(NO_PENDING_BROWSER_MODEL_LOAD))?;
        if pending.load_id != request.load_id
            || pending.model_id != request.model_id
            || pending.runtime_fingerprint != request.runtime_fingerprint
        {
            self.pending = Some(pending);
            return Err(invalid_source(BROWSER_LOAD_COMMIT_MISMATCH));
        }

        let loaded_at = now_iso();
        {
            let entry = self
                .manifest
                .models
                .get_mut(&request.model_id)
                .ok_or_else(|| model_not_found(&request.model_id))?;
            entry.last_loaded_at = Some(loaded_at.clone());
            entry.runtime_fingerprint = Some(request.runtime_fingerprint.clone());
            entry.updated_at = loaded_at;
        }
        validate_manifest(&self.manifest)?;
        let entry = self
            .manifest
            .models
            .get(&request.model_id)
            .ok_or_else(|| model_not_found(&request.model_id))?
            .clone();
        let model = self.model_info_from_entry(&entry);
        self.emit(
            BrowserObservabilityEventType::LoadComplete,
            SnapshotPatch {
                mode: None,
                state: Some(BrowserLifecycleState::Ready),
                model: Some(Some(model.clone())),
                query: None,
                runtime: Some(request.runtime),
                profile: Some(request.profile),
            },
        );
        let response = self.response_context();
        Ok(BrowserCommitLoadResponse {
            model,
            manifest: response.manifest,
            snapshot: response.snapshot,
            events: response.events,
        })
    }

    pub fn remove(
        &mut self,
        request: BrowserRemoveRequest,
    ) -> Result<BrowserRemoveResponse, ModelError> {
        if request.active_model_id.as_deref() == Some(request.model_id.as_str()) {
            return Err(ModelError::ModelInUse(request.model_id));
        }
        let model_id = request.model_id;
        let removed = self
            .manifest
            .models
            .remove(&model_id)
            .ok_or_else(|| model_not_found(&model_id))?;
        self.decrement_existing_refs(&removed);
        let orphaned_assets = self.remove_orphaned_assets();
        if contains_projector_asset(&orphaned_assets) {
            self.bump_projector_index_revision()?;
        }
        validate_manifest(&self.manifest)?;
        let response = self.response_context();
        Ok(BrowserRemoveResponse {
            removed,
            orphaned_assets,
            manifest: response.manifest,
            snapshot: response.snapshot,
            events: response.events,
        })
    }

    pub fn close(&mut self) -> BrowserObservabilitySnapshot {
        self.pending = None;
        self.pending_remote = None;
        self.emit(
            BrowserObservabilityEventType::Close,
            SnapshotPatch {
                mode: None,
                state: Some(BrowserLifecycleState::Closed),
                model: Some(None),
                query: Some(None),
                runtime: Some(None),
                profile: Some(None),
            },
        );
        self.snapshot.clone()
    }

    pub fn record_event(
        &mut self,
        event_type: BrowserObservabilityEventType,
        patch: Value,
    ) -> Result<BrowserObservabilitySnapshot, ModelError> {
        let patch = serde_json::from_value::<SnapshotPatch>(patch)?;
        self.emit(event_type, patch);
        Ok(self.snapshot.clone())
    }

    fn response_context(&mut self) -> BrowserResponseContext {
        BrowserResponseContext {
            manifest: self.manifest.clone(),
            snapshot: self.snapshot.clone(),
            events: self.drain_events(),
        }
    }

    fn upsert_assets(
        &mut self,
        assets: Vec<BrowserAssetRecord>,
        classified: &[ClassifiedAsset],
    ) -> Result<(), ModelError> {
        let mut projector_index_changed = false;
        let classified_by_id: BTreeMap<_, _> = classified
            .iter()
            .map(|asset| (asset.asset_id.as_str(), &asset.inspection))
            .collect();
        for mut asset in assets {
            validate_asset_record(&asset)?;
            let inspection = classified_by_id.get(asset.id.as_str()).copied();
            if let Some(inspection) = inspection {
                asset.inspection = Some((*inspection).clone());
                if inspection.role == crate::lifecycle::AssetRole::Projector {
                    asset.kind = ModelAssetKind::Projector;
                }
            }
            if let Some(existing) = self.manifest.assets.get(&asset.id) {
                let next_kind = if existing.kind == ModelAssetKind::Projector
                    || asset.kind == ModelAssetKind::Projector
                {
                    ModelAssetKind::Projector
                } else {
                    asset.kind
                };
                projector_index_changed |= existing.kind != next_kind
                    && (existing.kind == ModelAssetKind::Projector
                        || next_kind == ModelAssetKind::Projector);
                asset.ref_count = existing.ref_count;
                asset.created_at = existing.created_at.clone();
                asset.kind = next_kind;
                if asset.inspection.is_none() {
                    asset.inspection = existing.inspection.clone();
                }
            } else if asset.kind == ModelAssetKind::Projector {
                projector_index_changed = true;
            }
            self.manifest.assets.insert(asset.id.clone(), asset);
        }
        if projector_index_changed {
            self.bump_projector_index_revision()?;
        }
        Ok(())
    }

    fn upsert_base_model_entry(
        &mut self,
        plan: &PairingPlan,
    ) -> Result<BrowserModelEntry, ModelError> {
        let id = base_model_id(plan);
        let now = now_iso();
        let next_refs = sorted_model_asset_ids(&plan.model_asset_ids, None);
        let entry = if let Some(existing) = self.manifest.models.get(&id).cloned() {
            let previous_refs = entry_asset_ids(&existing);
            let mut updated = existing;
            updated.name = plan.name.clone();
            updated.model_asset_ids = plan.model_asset_ids.clone();
            if updated.projector_asset_id.is_none() {
                updated.modality = plan.modality;
                updated.status = plan.status;
            }
            updated.updated_at = now;
            self.rebalance_refs(&previous_refs, &entry_asset_ids(&updated))?;
            updated
        } else {
            let entry = BrowserModelEntry {
                id: id.clone(),
                name: plan.name.clone(),
                modality: plan.modality,
                status: plan.status,
                model_asset_ids: plan.model_asset_ids.clone(),
                projector_asset_id: None,
                pairing: None,
                runtime_fingerprint: None,
                created_at: now.clone(),
                updated_at: now,
                last_loaded_at: None,
            };
            self.increment_refs(&next_refs)?;
            entry
        };
        self.manifest.models.insert(id, entry.clone());
        validate_manifest(&self.manifest)?;
        Ok(entry)
    }

    fn derive_base_plan_for_entry(
        &self,
        entry: &BrowserModelEntry,
    ) -> Result<PairingPlan, ModelError> {
        let classified = self.map_model_assets(entry, |record| {
            classified_asset(
                record.id.clone(),
                record.name.clone(),
                record.inspection.clone(),
            )
        })?;
        PairingResolver::resolve(&classified)
    }

    fn resolve_entry_for_loading(
        &mut self,
        mut entry: BrowserModelEntry,
        base_plan: &PairingPlan,
    ) -> Result<BrowserModelEntry, ModelError> {
        let normalized_base_projector_types =
            normalize_projector_types(&base_plan.compatible_vision_projector_types);
        let normalized_base_audio_projector_types =
            normalize_projector_types(&base_plan.compatible_audio_projector_types);
        let normalized_base_audio_generation_projector_types =
            normalize_projector_types(&base_plan.compatible_audio_generation_projector_types);
        if let Some(projector_id) = entry.projector_asset_id.clone() {
            if let Some(projector) = self.manifest.assets.get(&projector_id).cloned() {
                if base_plan.compatible_vision_projector_types.is_empty()
                    && base_plan.compatible_audio_projector_types.is_empty()
                    && base_plan
                        .compatible_audio_generation_projector_types
                        .is_empty()
                {
                    return Ok(entry);
                }
                let inspection = projector.inspection.clone();
                if !projector_type_matches(inspection.as_ref(), base_plan) {
                    entry = self.detach_projector(&entry.id, base_plan)?;
                } else if match entry.pairing.as_ref() {
                    Some(pairing) => {
                        pairing.state != CoreModelPairingState::Resolved
                            || normalize_projector_types(&pairing.compatible_vision_projector_types)
                                != normalized_base_projector_types
                            || normalize_projector_types(&pairing.compatible_audio_projector_types)
                                != normalized_base_audio_projector_types
                            || normalize_projector_types(
                                &pairing.compatible_audio_generation_projector_types,
                            ) != normalized_base_audio_generation_projector_types
                    }
                    None => true,
                } {
                    entry = self.set_resolved_projector(&entry.id, &projector_id, base_plan)?;
                }
            } else {
                entry = self.detach_projector(&entry.id, base_plan)?;
            }
        }

        if base_plan.modality == ModelModality::Text {
            return self.set_unresolved_pairing(
                &entry.id,
                base_plan,
                ModelPairingReason::BaseNotMedia,
            );
        }
        if base_plan.compatible_vision_projector_types.is_empty()
            && base_plan.compatible_audio_projector_types.is_empty()
            && base_plan
                .compatible_audio_generation_projector_types
                .is_empty()
        {
            return self.set_unresolved_pairing(
                &entry.id,
                base_plan,
                ModelPairingReason::MissingMetadata,
            );
        }

        if entry.pairing.as_ref().is_some_and(|pairing| {
            pairing.state == CoreModelPairingState::Unresolved
                && pairing.checked_projector_index_revision
                    == self.manifest.projector_index_revision
                && normalize_projector_types(&pairing.compatible_vision_projector_types)
                    == normalized_base_projector_types
                && normalize_projector_types(&pairing.compatible_audio_projector_types)
                    == normalized_base_audio_projector_types
                && normalize_projector_types(&pairing.compatible_audio_generation_projector_types)
                    == normalized_base_audio_generation_projector_types
        }) {
            return Ok(entry);
        }

        let matches = self.find_compatible_installed_projector_ids(base_plan);
        if matches.len() == 1 {
            self.set_resolved_projector(&entry.id, &matches[0], base_plan)
        } else {
            self.set_unresolved_pairing(
                &entry.id,
                base_plan,
                if matches.is_empty() {
                    ModelPairingReason::NoMatch
                } else {
                    ModelPairingReason::MultipleMatches
                },
            )
        }
    }

    fn find_compatible_installed_projector_ids(&self, base_plan: &PairingPlan) -> Vec<String> {
        let mut matches = Vec::new();
        for asset in self.manifest.assets.values() {
            if asset.kind != ModelAssetKind::Projector || asset.ref_count == 0 {
                continue;
            }
            let inspection = asset.inspection.as_ref();
            if projector_type_matches(inspection, base_plan) {
                matches.push(asset.id.clone());
            }
        }
        sorted_values(matches)
    }

    fn set_resolved_projector(
        &mut self,
        id: &str,
        projector_asset_id: &str,
        plan: &PairingPlan,
    ) -> Result<BrowserModelEntry, ModelError> {
        let now = now_iso();
        let revision = self.manifest.projector_index_revision;
        self.update_model_entry(id, |entry| {
            entry.projector_asset_id = Some(projector_asset_id.to_string());
            entry.modality = plan.modality;
            entry.status = ModelStatus::Ready;
            entry.pairing = Some(browser_pairing(
                CoreModelPairingState::Resolved,
                revision,
                &plan.compatible_vision_projector_types,
                &plan.compatible_audio_projector_types,
                &plan.compatible_audio_generation_projector_types,
                None,
                &now,
            ));
            entry.updated_at = now;
        })
    }

    fn set_unresolved_pairing(
        &mut self,
        id: &str,
        plan: &PairingPlan,
        reason_code: ModelPairingReason,
    ) -> Result<BrowserModelEntry, ModelError> {
        let now = now_iso();
        let revision = self.manifest.projector_index_revision;
        self.update_model_entry(id, |entry| {
            entry.projector_asset_id = None;
            entry.modality = plan.modality;
            entry.status = plan.status;
            entry.pairing = Some(browser_pairing(
                CoreModelPairingState::Unresolved,
                revision,
                &plan.compatible_vision_projector_types,
                &plan.compatible_audio_projector_types,
                &plan.compatible_audio_generation_projector_types,
                Some(reason_code),
                &now,
            ));
            entry.updated_at = now;
        })
    }

    fn detach_projector(
        &mut self,
        id: &str,
        base_plan: &PairingPlan,
    ) -> Result<BrowserModelEntry, ModelError> {
        let now = now_iso();
        self.update_model_entry(id, |entry| {
            entry.projector_asset_id = None;
            entry.modality = base_plan.modality;
            entry.status = base_plan.status;
            entry.pairing = None;
            entry.updated_at = now;
        })
    }

    fn update_model_entry(
        &mut self,
        id: &str,
        update: impl FnOnce(&mut BrowserModelEntry),
    ) -> Result<BrowserModelEntry, ModelError> {
        let mut entry = self
            .manifest
            .models
            .get(id)
            .cloned()
            .ok_or_else(|| model_not_found(id))?;
        let previous_refs = entry_asset_ids(&entry);
        update(&mut entry);
        let next_refs = entry_asset_ids(&entry);
        self.rebalance_refs(&previous_refs, &next_refs)?;
        self.manifest.models.insert(id.to_string(), entry.clone());
        validate_manifest(&self.manifest)?;
        Ok(entry)
    }

    fn planned_assets_for_entry(
        &self,
        entry: &BrowserModelEntry,
    ) -> Result<(Vec<BrowserPlannedAsset>, Option<BrowserPlannedAsset>), ModelError> {
        let assets = self.map_model_assets(entry, planned_asset)?;
        let projector = entry
            .projector_asset_id
            .as_deref()
            .map(|asset_id| {
                let record = self
                    .manifest
                    .assets
                    .get(asset_id)
                    .ok_or_else(|| missing_model_asset(asset_id))?;
                Ok::<BrowserPlannedAsset, ModelError>(planned_asset(record))
            })
            .transpose()?;
        Ok((assets, projector))
    }

    fn map_model_assets<T>(
        &self,
        entry: &BrowserModelEntry,
        mut map: impl FnMut(&BrowserAssetRecord) -> T,
    ) -> Result<Vec<T>, ModelError> {
        entry
            .model_asset_ids
            .iter()
            .map(|asset_id| {
                self.manifest
                    .assets
                    .get(asset_id)
                    .ok_or_else(|| missing_model_asset(asset_id))
                    .map(&mut map)
            })
            .collect()
    }

    fn model_info_from_entry(&self, entry: &BrowserModelEntry) -> BrowserModelInfo {
        let assets = entry_asset_ids(entry)
            .into_iter()
            .filter_map(|asset_id| self.manifest.assets.get(&asset_id));
        let summary = browser_asset_summary(assets);
        BrowserModelInfo {
            id: entry.id.clone(),
            name: entry.name.clone(),
            modality: entry.modality,
            status: entry.status,
            source: summary.source,
            bytes: summary.bytes,
            asset_fingerprint: asset_fingerprint(entry),
            created_at: entry.created_at.clone(),
            updated_at: entry.updated_at.clone(),
        }
    }

    fn increment_refs(&mut self, asset_ids: &[String]) -> Result<(), ModelError> {
        self.adjust_refs(asset_ids, increment_asset_refcount)
    }

    fn decrement_refs(&mut self, asset_ids: &[String]) -> Result<(), ModelError> {
        self.adjust_refs(asset_ids, decrement_asset_refcount)
    }

    fn adjust_refs(
        &mut self,
        asset_ids: &[String],
        adjust_refcount: fn(&mut u32, &str) -> Result<(), ModelError>,
    ) -> Result<(), ModelError> {
        for id in sorted_unique_strings(asset_ids.to_vec()) {
            let asset = self
                .manifest
                .assets
                .get_mut(&id)
                .ok_or_else(|| missing_model_asset(&id))?;
            adjust_refcount(&mut asset.ref_count, &id)?;
        }
        Ok(())
    }

    fn decrement_existing_refs(&mut self, entry: &BrowserModelEntry) {
        for asset_id in entry_asset_ids(entry) {
            let Some(asset) = self.manifest.assets.get_mut(&asset_id) else {
                continue;
            };
            if asset.ref_count > 0 {
                asset.ref_count -= 1;
            }
        }
    }

    fn remove_orphaned_assets(&mut self) -> Vec<BrowserAssetRecord> {
        remove_matching_values(&mut self.manifest.assets, |asset| asset.ref_count == 0)
    }

    fn rebalance_refs(
        &mut self,
        previous_refs: &[String],
        next_refs: &[String],
    ) -> Result<(), ModelError> {
        let (removed, added) = sorted_ref_deltas(previous_refs, next_refs);
        self.decrement_refs(&removed)?;
        self.increment_refs(&added)
    }

    fn bump_projector_index_revision(&mut self) -> Result<(), ModelError> {
        bump_revision(&mut self.manifest.projector_index_revision)
    }

    fn emit(&mut self, event_type: BrowserObservabilityEventType, patch: SnapshotPatch) {
        self.snapshot = apply_snapshot_patch(self.snapshot.clone(), patch);
        let event = BrowserObservabilityEvent {
            event_type,
            snapshot: self.snapshot.clone(),
        };
        self.events.push_back(event);
    }
}

fn accept_remote_payload(
    pending: &mut PendingRemoteAcquisition,
    manifest: &BrowserRegistryManifest,
    action: &RemoteAction,
    event: &RemoteAcquisitionEvent,
    assets: Vec<BrowserAssetRecord>,
    classified: Vec<ClassifiedAsset>,
) -> Result<(), ModelError> {
    match (action, event) {
        (
            RemoteAction::Download { metadata, .. },
            RemoteAcquisitionEvent::DownloadSucceeded {
                asset_ids,
                created_asset_ids,
                ..
            },
        ) => {
            validate_remote_download_assets(metadata, asset_ids, created_asset_ids, &assets)?;
            insert_remote_assets(pending, assets, classified)
        }
        (
            RemoteAction::ValidateCache { candidate, .. },
            RemoteAcquisitionEvent::CacheValidated { asset_ids, .. },
        ) => {
            if asset_ids != &candidate.asset_ids {
                return Err(ModelError::RemoteIntegrityFailed {
                    url: candidate.metadata.url.clone(),
                    reason: "cache validation returned different asset identifiers".to_string(),
                });
            }
            let cached = asset_ids
                .iter()
                .map(|asset_id| {
                    manifest
                        .assets
                        .get(asset_id)
                        .cloned()
                        .ok_or_else(|| missing_model_asset(asset_id))
                })
                .collect::<Result<Vec<_>, _>>()?;
            insert_remote_assets(pending, cached, classified)
        }
        (
            RemoteAction::Cleanup { asset_ids, .. },
            RemoteAcquisitionEvent::CleanupSucceeded { .. },
        ) => {
            for asset_id in asset_ids {
                pending.assets.remove(asset_id);
                pending.classified.remove(asset_id);
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn insert_remote_assets(
    pending: &mut PendingRemoteAcquisition,
    assets: Vec<BrowserAssetRecord>,
    classified: Vec<ClassifiedAsset>,
) -> Result<(), ModelError> {
    let mut classified_by_id = BTreeMap::new();
    for asset in classified {
        if classified_by_id
            .insert(asset.asset_id.clone(), asset)
            .is_some()
        {
            return Err(invalid_source(
                "remote asset classification contains duplicate identifiers",
            ));
        }
    }
    let mut prepared = Vec::with_capacity(assets.len());
    for record in assets {
        validate_asset_record(&record)?;
        let classified = classified_by_id
            .remove(&record.id)
            .or_else(|| {
                record.inspection.clone().map(|inspection| ClassifiedAsset {
                    asset_id: record.id.clone(),
                    name: record.name.clone(),
                    inspection,
                })
            })
            .ok_or_else(|| invalid_source("remote asset classification is missing"))?;
        if classified.name != record.name {
            return Err(invalid_source(
                "remote asset classification name does not match its record",
            ));
        }
        if prepared.iter().any(
            |(prepared_record, _): &(BrowserAssetRecord, ClassifiedAsset)| {
                prepared_record.id == record.id
            },
        ) {
            return Err(invalid_source(
                "remote asset receipt contains duplicate identifiers",
            ));
        }
        prepared.push((record, classified));
    }
    if !classified_by_id.is_empty() {
        return Err(invalid_source(
            "remote classification references an unreported asset",
        ));
    }
    for (record, classified) in prepared {
        pending.classified.insert(record.id.clone(), classified);
        pending.assets.insert(record.id.clone(), record);
    }
    Ok(())
}

fn validate_remote_download_assets(
    metadata: &RemoteMetadata,
    asset_ids: &[String],
    created_asset_ids: &[String],
    assets: &[BrowserAssetRecord],
) -> Result<(), ModelError> {
    let reported_ids: Vec<_> = assets.iter().map(|asset| asset.id.clone()).collect();
    if reported_ids != asset_ids {
        return Err(remote_record_error(
            metadata,
            "download receipts do not match the reported asset identifiers",
        ));
    }
    if created_asset_ids
        .iter()
        .any(|asset_id| !asset_ids.contains(asset_id))
    {
        return Err(remote_record_error(
            metadata,
            "created asset identifiers are not part of the download receipt",
        ));
    }
    for asset in assets {
        if asset.source_url.as_deref() != Some(metadata.url.as_str())
            || asset.source_etag != metadata.etag
            || asset.source_last_modified != metadata.last_modified
            || asset.source_bytes != Some(metadata.bytes)
        {
            return Err(remote_record_error(
                metadata,
                "download receipt does not match remote metadata",
            ));
        }
    }
    validate_remote_asset_layout(metadata, assets)
}

fn validate_remote_asset_layout(
    metadata: &RemoteMetadata,
    assets: &[BrowserAssetRecord],
) -> Result<(), ModelError> {
    if assets.len() == 1 {
        return Ok(());
    }
    if !assets
        .iter()
        .all(|asset| asset.kind == ModelAssetKind::Shard)
    {
        return Err(remote_record_error(
            metadata,
            "download receipt has an invalid asset layout",
        ));
    }
    validate_complete_browser_split(assets).map_err(|reason| remote_record_error(metadata, reason))
}

fn validate_complete_browser_split(assets: &[BrowserAssetRecord]) -> Result<(), &'static str> {
    let count = u32::try_from(assets.len()).map_err(|_| "split contains too many assets")?;
    let mut indices: Vec<_> = assets
        .iter()
        .map(|asset| {
            if asset.source_part_count != Some(count) {
                return Err("split asset count is inconsistent");
            }
            asset
                .source_part_index
                .ok_or("split asset index is missing")
        })
        .collect::<Result<_, _>>()?;
    indices.sort_unstable();
    if indices.iter().copied().eq(0..count) {
        Ok(())
    } else {
        Err("split asset indices are incomplete")
    }
}

fn remote_record_error(metadata: &RemoteMetadata, reason: &str) -> ModelError {
    ModelError::RemoteIntegrityFailed {
        url: metadata.url.clone(),
        reason: reason.to_string(),
    }
}

fn browser_remote_requests(
    manifest: &BrowserRegistryManifest,
    urls: impl IntoIterator<Item = String>,
) -> Result<Vec<RemoteAcquisitionRequest>, ModelError> {
    urls.into_iter()
        .enumerate()
        .map(|(index, url)| {
            let url = canonical_remote_url(&url)?;
            Ok(RemoteAcquisitionRequest {
                member_id: u32::try_from(index)
                    .map_err(|_| invalid_source("remote model contains too many source members"))?,
                candidates: browser_remote_candidates(manifest, &url)?,
                url,
            })
        })
        .collect()
}

fn browser_remote_candidates(
    manifest: &BrowserRegistryManifest,
    url: &str,
) -> Result<Vec<RemoteCacheCandidate>, ModelError> {
    type CandidateKey = (u8, u64, Option<String>, Option<String>);
    let mut groups: BTreeMap<CandidateKey, Vec<&BrowserAssetRecord>> = BTreeMap::new();
    for asset in manifest.assets.values() {
        let Some(source_url) = asset.source_url.as_deref() else {
            continue;
        };
        let source_url = canonical_remote_url(source_url)?;
        if source_url != url {
            continue;
        }
        let layout = u8::from(asset.source_part_count.is_some());
        groups
            .entry((
                layout,
                asset.source_bytes.unwrap_or(asset.bytes),
                asset.source_etag.clone(),
                asset.source_last_modified.clone(),
            ))
            .or_default()
            .push(asset);
    }
    groups
        .into_iter()
        .map(|((layout, bytes, etag, last_modified), mut records)| {
            records.sort_by_key(|record| record.source_part_index.unwrap_or(0));
            if layout == 1 {
                validate_complete_browser_split(
                    &records
                        .iter()
                        .map(|record| (*record).clone())
                        .collect::<Vec<_>>(),
                )
                .map_err(|reason| ModelError::StorageCorrupt(reason.to_string()))?;
            } else if records.len() != 1 {
                return Err(ModelError::StorageCorrupt(
                    "remote cache identity maps to multiple unsplit assets".to_string(),
                ));
            }
            let asset_ids: Vec<_> = records.iter().map(|record| record.id.clone()).collect();
            let name = records
                .first()
                .map(|record| record.name.clone())
                .ok_or_else(|| ModelError::StorageCorrupt("empty cache candidate".to_string()))?;
            Ok(RemoteCacheCandidate {
                candidate_id: sha256_hex(asset_ids.join("\n").as_bytes()),
                asset_ids,
                metadata: RemoteMetadata {
                    url: url.to_string(),
                    name,
                    bytes,
                    etag,
                    last_modified,
                },
            })
        })
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SnapshotPatch {
    #[serde(default)]
    mode: Option<BrowserObservabilityMode>,
    #[serde(default)]
    state: Option<BrowserLifecycleState>,
    #[serde(default)]
    model: Option<Option<BrowserModelInfo>>,
    #[serde(default)]
    query: Option<Option<BrowserQueryObservation>>,
    #[serde(default)]
    runtime: Option<Option<Value>>,
    #[serde(default)]
    profile: Option<Option<Value>>,
}

pub fn success_response<T>(value: T) -> BrowserLifecycleEnvelope<T>
where
    T: Serialize,
{
    BrowserLifecycleEnvelope {
        ok: true,
        value: Some(value),
        error: None,
    }
}

pub fn error_response<T>(error: ModelError) -> BrowserLifecycleEnvelope<T>
where
    T: Serialize,
{
    BrowserLifecycleEnvelope {
        ok: false,
        value: None,
        error: Some(lifecycle_error(error)),
    }
}

pub fn response_json<T>(response: BrowserLifecycleEnvelope<T>) -> String
where
    T: Serialize,
{
    serde_json::to_string(&response)
        .unwrap_or_else(|_| LIFECYCLE_SERIALIZATION_FALLBACK.to_string())
}

fn lifecycle_error(error: ModelError) -> BrowserLifecycleError {
    BrowserLifecycleError {
        code: error.code(),
        status: error.status(),
        retry_after_ms: error.retry_after_ms(),
        message: error.to_string(),
    }
}

fn validate_manifest(manifest: &BrowserRegistryManifest) -> Result<(), ModelError> {
    validate_registry_manifest_version(BROWSER_REGISTRY_MANIFEST_LABEL, manifest.version)?;
    let mut expected_ref_counts = BTreeMap::<String, u32>::new();
    for (id, asset) in &manifest.assets {
        if id != &asset.id {
            return Err(manifest_key_mismatch("asset", id, &asset.id));
        }
        validate_asset_record(asset)?;
    }
    for (id, model) in &manifest.models {
        if id != &model.id {
            return Err(manifest_key_mismatch("model", id, &model.id));
        }
        for asset_id in entry_asset_ids(model) {
            if !manifest.assets.contains_key(&asset_id) {
                return Err(model_missing_asset(id, &asset_id));
            }
            increment_expected_asset_refcount(&mut expected_ref_counts, &asset_id)?;
        }
    }
    for (id, asset) in &manifest.assets {
        let expected = expected_ref_counts.get(id).copied().unwrap_or(0);
        if asset.ref_count != expected {
            return Err(asset_refcount_mismatch(id, asset.ref_count, expected));
        }
    }
    Ok(())
}

fn validate_asset_record(asset: &BrowserAssetRecord) -> Result<(), ModelError> {
    if asset.id.trim().is_empty() {
        return Err(empty_asset_id());
    }
    if asset.storage_path.trim().is_empty() {
        return Err(invalid_asset_field(
            &asset.id,
            "storagePath must not be empty",
        ));
    }
    if asset.bytes == 0 {
        return Err(invalid_asset_field(&asset.id, "byte size must be positive"));
    }
    if let Some(inspection) = &asset.inspection {
        validate_asset_inspection_version(inspection.version)?;
    }
    let has_split_index = asset.source_part_index.is_some() || asset.source_part_count.is_some();
    if has_split_index {
        let index = asset
            .source_part_index
            .ok_or_else(|| invalid_asset_field(&asset.id, "split part index is missing"))?;
        let count = asset
            .source_part_count
            .ok_or_else(|| invalid_asset_field(&asset.id, "split part count is missing"))?;
        if count == 0 || index >= count {
            return Err(invalid_asset_field(
                &asset.id,
                "split part index/count is invalid",
            ));
        }
    }
    Ok(())
}

fn planned_asset(record: &BrowserAssetRecord) -> BrowserPlannedAsset {
    BrowserPlannedAsset {
        asset_id: record.id.clone(),
        kind: record.kind,
        storage_path: record.storage_path.clone(),
        mount_name: record.name.clone(),
        bytes: record.bytes,
    }
}

fn contains_projector_asset(assets: &[BrowserAssetRecord]) -> bool {
    assets
        .iter()
        .any(|asset| asset.kind == ModelAssetKind::Projector)
}

fn browser_asset_summary<'asset>(
    assets: impl Iterator<Item = &'asset BrowserAssetRecord>,
) -> AssetSummary {
    asset_summary(assets.map(|asset| (asset.bytes, asset.source_url.is_some())))
}

fn entry_asset_ids(entry: &BrowserModelEntry) -> Vec<String> {
    sorted_model_asset_ids(&entry.model_asset_ids, entry.projector_asset_id.as_ref())
}

fn normalize_projector_types(projector_types: &[String]) -> Vec<String> {
    sorted_unique_strings(projector_types.to_vec())
}

fn projector_type_matches(inspection: Option<&AssetInspection>, plan: &PairingPlan) -> bool {
    let Some(inspection) = inspection else {
        return false;
    };
    projector_role_matches(
        &plan.compatible_vision_projector_types,
        inspection.provided_vision_projector_type.as_ref(),
    ) && projector_role_matches(
        &plan.compatible_audio_projector_types,
        inspection.provided_audio_projector_type.as_ref(),
    ) && projector_role_matches(
        &plan.compatible_audio_generation_projector_types,
        inspection.provided_audio_generation_projector_type.as_ref(),
    )
}

fn projector_role_matches(required: &[String], provided: Option<&String>) -> bool {
    required.is_empty()
        || provided.is_some_and(|provided| required.iter().any(|value| value == provided))
}

fn browser_pairing(
    state: CoreModelPairingState,
    projector_index_revision: u64,
    compatible_vision_projector_types: &[String],
    compatible_audio_projector_types: &[String],
    compatible_audio_generation_projector_types: &[String],
    reason_code: Option<ModelPairingReason>,
    updated_at: &str,
) -> BrowserModelPairing {
    BrowserModelPairing {
        state,
        checked_projector_index_revision: projector_index_revision,
        compatible_vision_projector_types: normalize_projector_types(
            compatible_vision_projector_types,
        ),
        compatible_audio_projector_types: normalize_projector_types(
            compatible_audio_projector_types,
        ),
        compatible_audio_generation_projector_types: normalize_projector_types(
            compatible_audio_generation_projector_types,
        ),
        reason_code,
        updated_at: updated_at.to_string(),
    }
}

fn base_model_id(plan: &PairingPlan) -> String {
    let hash = stable_json_hash(&json!({
        "modelAssetIds": sorted_model_asset_ids(&plan.model_asset_ids, None),
    }));
    format!("model-{}", &hash[..BROWSER_MODEL_ID_HASH_CHARS])
}

fn asset_fingerprint(entry: &BrowserModelEntry) -> String {
    stable_json_hash(&json!({
        "modelAssetIds": sorted_model_asset_ids(&entry.model_asset_ids, None),
        "projectorAssetId": entry.projector_asset_id,
    }))
}

fn browser_load_id(model_id: &str, asset_fingerprint: &str, runtime_fingerprint: &str) -> String {
    stable_json_hash(&json!({
        "modelId": model_id,
        "assetFingerprint": asset_fingerprint,
        "runtimeFingerprint": runtime_fingerprint,
        "nonce": now_iso(),
    }))
}

fn browser_backend_plan(options: &BrowserLoadOptions) -> Result<BackendPlan, ModelError> {
    browser_backend_plan_with_capabilities(options, None)
}

fn browser_backend_plan_with_capabilities(
    options: &BrowserLoadOptions,
    capabilities: Option<&BackendCapabilities>,
) -> Result<BackendPlan, ModelError> {
    let backend = match options.backend {
        BrowserBackendPreference::Auto => BackendPreference::Auto,
        BrowserBackendPreference::Cpu => BackendPreference::Cpu,
        BrowserBackendPreference::WebGpu => BackendPreference::WebGpu,
    };
    let load_options = ModelLoadOptions {
        backend,
        stats: stats_mode(options.observability),
        runtime: browser_runtime_config(&options.runtime)?,
    };
    if let Some(capabilities) = capabilities {
        return BackendPolicy::select_with_capabilities(&load_options, capabilities);
    }

    let selected = match options.backend {
        BrowserBackendPreference::Auto | BrowserBackendPreference::Cpu => {
            BackendPreference::Cpu.as_str()
        }
        BrowserBackendPreference::WebGpu => BackendPreference::WebGpu.as_str(),
    };
    Ok(BackendPolicy::select_known(
        &load_options,
        selected,
        vec![selected.to_string()],
        Some(format!("browser selected {selected}")),
    ))
}

fn browser_runtime_config(runtime: &Value) -> Result<NativeRuntimeConfig, ModelError> {
    let runtime = if runtime.is_null() {
        json!({})
    } else {
        runtime.clone()
    };
    Ok(serde_json::from_value(runtime)?)
}

fn stats_mode(mode: BrowserObservabilityMode) -> StatsMode {
    match mode {
        BrowserObservabilityMode::Off => StatsMode::Off,
        BrowserObservabilityMode::Runtime => StatsMode::Basic,
        BrowserObservabilityMode::Profile => StatsMode::Profile,
    }
}

fn runtime_fingerprint(runtime_config: &Value, backend: &BackendSelection) -> String {
    stable_json_hash(&json!({
        "backend": backend.selected,
        "runtime": runtime_config,
    }))
}

fn apply_snapshot_patch(
    mut snapshot: BrowserObservabilitySnapshot,
    patch: SnapshotPatch,
) -> BrowserObservabilitySnapshot {
    if let Some(mode) = patch.mode {
        snapshot.mode = mode;
    }
    if let Some(state) = patch.state {
        snapshot.state = state;
    }
    if let Some(model) = patch.model {
        snapshot.model = model;
    }
    if let Some(query) = patch.query {
        snapshot.query = query;
    }
    if let Some(runtime) = patch.runtime {
        snapshot.runtime = runtime;
    }
    if let Some(profile) = patch.profile {
        snapshot.profile = profile;
    }
    snapshot.updated_at = now_iso();
    snapshot
}

fn stable_json_hash(value: &Value) -> String {
    sha256_hex(stable_json(value).as_bytes())
}

fn stable_json(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("string serialization"),
        Value::Array(values) => {
            let inner = values.iter().map(stable_json).collect::<Vec<_>>().join(",");
            format!("[{inner}]")
        }
        Value::Object(values) => {
            let inner = values
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("key serialization"),
                        stable_json(value)
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{inner}}}")
        }
    }
}

fn now_iso() -> String {
    let ms = super::storage::now_unix_ms();
    iso_from_unix_ms(ms)
}

fn iso_from_unix_ms(ms: u64) -> String {
    let seconds = ms / MILLIS_PER_SECOND;
    let millis = ms % MILLIS_PER_SECOND;
    let days = (seconds / SECONDS_PER_DAY) as i64;
    let seconds_of_day = seconds % SECONDS_PER_DAY;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / SECONDS_PER_HOUR;
    let minute = (seconds_of_day % SECONDS_PER_HOUR) / SECONDS_PER_MINUTE;
    let second = seconds_of_day % SECONDS_PER_MINUTE;
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}Z")
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i64, u32, u32) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = mp + if mp < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }
    (year, month as u32, day as u32)
}

#[cfg(test)]
#[path = "../tests/lifecycle/browser_tests.rs"]
mod browser_tests;
