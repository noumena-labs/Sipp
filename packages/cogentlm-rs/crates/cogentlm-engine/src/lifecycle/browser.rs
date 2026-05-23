//! Browser lifecycle catalog.
//!
//! This module owns the browser-facing model registry contract without taking a
//! dependency on OPFS, `File`, `fetch`, or WORKERFS. The browser host installs
//! assets and mounts files; Rust owns the persisted lifecycle decisions.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::collection::{
    remove_matching_values, sorted_ref_deltas, sorted_unique_strings, sorted_values,
};
use crate::lifecycle::util::{
    asset_refcount_mismatch, asset_summary, bump_projector_index_revision as bump_revision,
    classified_asset, decrement_asset_refcount, empty_asset_id, increment_asset_refcount,
    increment_expected_asset_refcount, invalid_asset_field, invalid_pairing, invalid_source,
    manifest_key_mismatch, media_marker_for_modality, missing_model_asset, model_missing_asset,
    model_not_found, sha256_hex, sorted_model_asset_ids, validate_registry_manifest_version,
    AssetSummary,
};
use crate::lifecycle::{
    AssetInspection, ClassifiedAsset, ModelAssetKind, ModelError, ModelModality,
    ModelPairingReason, ModelPairingState as CoreModelPairingState, ModelSourceKind, ModelStatus,
    PairingPlan, PairingResolver, REGISTRY_MANIFEST_VERSION,
};
use crate::runtime::numeric::{
    MILLIS_PER_SECOND, SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE,
};

const EXPLICIT_PROJECTOR_MISSING_ID: &str = "explicit projector did not produce a projector id";
const NO_PENDING_BROWSER_MODEL_LOAD: &str = "no pending browser model load";
const BROWSER_LOAD_COMMIT_MISMATCH: &str =
    "browser model load commit does not match the pending load";
const BROWSER_REGISTRY_MANIFEST_LABEL: &str = "browser registry manifest";
const CODE_INVALID_MODEL_SOURCE: &str = "INVALID_MODEL_SOURCE";
const CODE_INVALID_MODEL_PAIRING: &str = "INVALID_MODEL_PAIRING";
const CODE_STORAGE_UNAVAILABLE: &str = "STORAGE_UNAVAILABLE";
const CODE_STORAGE_CORRUPT: &str = "STORAGE_CORRUPT";
const CODE_MODEL_BROKEN: &str = "MODEL_BROKEN";
const CODE_MODEL_NOT_FOUND: &str = "MODEL_NOT_FOUND";
const CODE_REMOTE_LOAD_FAILED: &str = "REMOTE_LOAD_FAILED";
const CODE_QUERY_FAILED: &str = "QUERY_FAILED";
const QUERY_STATUS_FAILED: &str = "failed";
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
    pub hash: String,
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
    #[serde(default)]
    pub compatible_vision_projector_types: Vec<String>,
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
    pub loaded: bool,
    pub chat_template: Option<String>,
    pub bos_text: String,
    pub eos_text: String,
    pub media_marker: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserLoadedModelState {
    pub id: String,
    pub asset_fingerprint: String,
    pub runtime_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
    #[serde(default)]
    pub bos_text: String,
    #[serde(default)]
    pub eos_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_marker: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserQueryObservation {
    pub session: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserLoadOptions {
    #[serde(default)]
    pub runtime: Value,
    #[serde(default)]
    pub observability: BrowserObservabilityMode,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BrowserLoadSource {
    Installed {
        id: String,
        #[serde(default)]
        classified_projectors: Vec<ClassifiedAsset>,
    },
    Assets {
        assets: Vec<BrowserAssetRecord>,
        classified: Vec<ClassifiedAsset>,
        #[serde(default)]
        explicit_projector_asset_id: Option<String>,
        #[serde(default)]
        classified_projectors: Vec<ClassifiedAsset>,
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
    pub load_required: bool,
    #[serde(default)]
    pub assets: Vec<BrowserPlannedAsset>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector: Option<BrowserPlannedAsset>,
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
    pub chat_template: Option<String>,
    #[serde(default)]
    pub bos_text: String,
    #[serde(default)]
    pub eos_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_marker: Option<String>,
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
}

#[derive(Debug, Clone)]
struct PendingLoad {
    load_id: String,
    model_id: String,
    runtime_fingerprint: String,
}

struct BrowserResponseContext {
    manifest: BrowserRegistryManifest,
    snapshot: BrowserObservabilitySnapshot,
    events: Vec<BrowserObservabilityEvent>,
}

#[derive(Debug, Clone)]
pub struct BrowserLifecycleService {
    pub manifest: BrowserRegistryManifest,
    current: Option<BrowserLoadedModelState>,
    pending: Option<PendingLoad>,
    pub snapshot: BrowserObservabilitySnapshot,
    events: VecDeque<BrowserObservabilityEvent>,
}

impl BrowserLifecycleService {
    pub fn create(config: BrowserCreateConfig) -> Result<Self, ModelError> {
        let manifest = migrate_manifest(config.manifest.unwrap_or_default())?;
        validate_manifest(&manifest)?;
        let now = now_iso();
        Ok(Self {
            manifest,
            current: None,
            pending: None,
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

    pub fn current(&self) -> Option<BrowserModelInfo> {
        self.current.as_ref().and_then(|current| {
            self.manifest
                .models
                .get(&current.id)
                .map(|entry| self.model_info_from_entry(entry))
        })
    }

    pub fn drain_events(&mut self) -> Vec<BrowserObservabilityEvent> {
        self.events.drain(..).collect()
    }

    pub fn prepare_load(
        &mut self,
        source: BrowserLoadSource,
        options: BrowserLoadOptions,
    ) -> Result<BrowserPrepareLoadResponse, ModelError> {
        validate_manifest(&self.manifest)?;
        let entry = match source {
            BrowserLoadSource::Installed {
                id,
                classified_projectors,
            } => {
                let entry = self
                    .manifest
                    .models
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| model_not_found(&id))?;
                let base_plan = self.derive_base_plan_for_entry(&entry)?;
                self.resolve_entry_for_loading(entry, &base_plan, &classified_projectors)?
            }
            BrowserLoadSource::Assets {
                assets,
                classified,
                explicit_projector_asset_id,
                classified_projectors,
            } => {
                self.upsert_assets(assets, &classified)?;
                let source_projector = resolve_source_projector_asset_id(
                    &classified,
                    explicit_projector_asset_id.as_deref(),
                );
                let base_classified: Vec<_> = classified
                    .iter()
                    .filter(|asset| Some(asset.asset_id.as_str()) != source_projector.as_deref())
                    .cloned()
                    .collect();
                let base_plan = PairingResolver::resolve(&base_classified)?;
                let mut entry = self.upsert_base_model_entry(&base_plan, &options)?;

                if let Some(projector_id) = source_projector {
                    let previous = entry.clone();
                    match PairingResolver::resolve_explicit(&classified, &projector_id) {
                        Ok(plan) => {
                            let projector = plan
                                .projector_asset_id
                                .clone()
                                .ok_or_else(|| invalid_pairing(EXPLICIT_PROJECTOR_MISSING_ID))?;
                            entry = self.set_resolved_projector(
                                &entry.id,
                                &projector,
                                &plan.compatible_vision_projector_types,
                            )?;
                        }
                        Err(error) => {
                            self.restore_entry(previous)?;
                            return Err(error);
                        }
                    }
                } else {
                    entry =
                        self.resolve_entry_for_loading(entry, &base_plan, &classified_projectors)?;
                }
                validate_manifest(&self.manifest)?;
                entry
            }
        };

        let runtime_config =
            runtime_config_with_observability(options.runtime, options.observability);
        let runtime_fingerprint = runtime_fingerprint(&runtime_config, options.observability);
        let asset_fingerprint = asset_fingerprint(&entry);
        let load_id = browser_load_id(&entry.id, &asset_fingerprint, &runtime_fingerprint);
        let load_required = browser_load_required(
            self.current.as_ref(),
            &entry.id,
            &asset_fingerprint,
            &runtime_fingerprint,
        );

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
            load_required,
            assets,
            projector,
            manifest: response.manifest,
            snapshot: response.snapshot,
            events: response.events,
        })
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
        self.current = Some(BrowserLoadedModelState {
            id: request.model_id,
            asset_fingerprint: asset_fingerprint(&entry),
            runtime_fingerprint: request.runtime_fingerprint,
            chat_template: request.chat_template,
            bos_text: request.bos_text,
            eos_text: request.eos_text,
            media_marker: request.media_marker,
        });
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

    pub fn abort_load(&mut self, message: Option<String>) -> BrowserObservabilitySnapshot {
        self.pending = None;
        self.emit(
            BrowserObservabilityEventType::Error,
            SnapshotPatch {
                mode: None,
                state: Some(BrowserLifecycleState::Error),
                model: None,
                query: Some(Some(failed_query_observation(message))),
                runtime: None,
                profile: None,
            },
        );
        self.snapshot.clone()
    }

    pub fn remove(&mut self, model_id: &str) -> Result<BrowserRemoveResponse, ModelError> {
        let removed = self
            .manifest
            .models
            .remove(model_id)
            .ok_or_else(|| model_not_found(model_id))?;
        self.decrement_existing_refs(&removed);
        let orphaned_assets = self.remove_orphaned_assets();
        if self
            .current
            .as_ref()
            .is_some_and(|current| current.id == model_id)
        {
            self.current = None;
        }
        if contains_projector_asset(&orphaned_assets) {
            self.bump_projector_index_revision()?;
        }
        validate_manifest(&self.manifest)?;
        let state = if self.current.is_some() {
            BrowserLifecycleState::Ready
        } else {
            BrowserLifecycleState::Idle
        };
        self.emit(
            BrowserObservabilityEventType::LoadComplete,
            SnapshotPatch {
                mode: None,
                state: Some(state),
                model: Some(self.current()),
                query: Some(None),
                runtime: Some(None),
                profile: Some(None),
            },
        );
        let response = self.response_context();
        Ok(BrowserRemoveResponse {
            removed,
            orphaned_assets,
            manifest: response.manifest,
            snapshot: response.snapshot,
            events: response.events,
        })
    }

    pub fn unload(&mut self) -> BrowserObservabilitySnapshot {
        self.current = None;
        self.pending = None;
        self.emit(
            BrowserObservabilityEventType::LoadComplete,
            SnapshotPatch {
                mode: None,
                state: Some(BrowserLifecycleState::Idle),
                model: Some(None),
                query: Some(None),
                runtime: Some(None),
                profile: Some(None),
            },
        );
        self.snapshot.clone()
    }

    pub fn close(&mut self) -> BrowserObservabilitySnapshot {
        self.current = None;
        self.pending = None;
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
        options: &BrowserLoadOptions,
    ) -> Result<BrowserModelEntry, ModelError> {
        let id = base_model_id(plan);
        let now = now_iso();
        let runtime_config =
            runtime_config_with_observability(options.runtime.clone(), options.observability);
        let runtime_fingerprint = runtime_fingerprint(&runtime_config, options.observability);
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
            updated.runtime_fingerprint = Some(runtime_fingerprint);
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
                runtime_fingerprint: Some(runtime_fingerprint),
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
        classified_projectors: &[ClassifiedAsset],
    ) -> Result<BrowserModelEntry, ModelError> {
        let normalized_base_projector_types =
            normalize_projector_types(&base_plan.compatible_vision_projector_types);
        let compatible_projector_types =
            compatible_projector_type_set(&base_plan.compatible_vision_projector_types);
        if let Some(projector_id) = entry.projector_asset_id.clone() {
            if let Some(projector) = self.manifest.assets.get(&projector_id).cloned() {
                if base_plan.compatible_vision_projector_types.is_empty() {
                    return Ok(entry);
                }
                let inspection = projector.inspection.clone().or_else(|| {
                    classified_projectors
                        .iter()
                        .find(|asset| asset.asset_id == projector_id)
                        .map(|asset| asset.inspection.clone())
                });
                if !projector_type_matches(inspection.as_ref(), &compatible_projector_types) {
                    entry = self.detach_projector(&entry.id, base_plan)?;
                } else if match entry.pairing.as_ref() {
                    Some(pairing) => {
                        pairing.state != CoreModelPairingState::Resolved
                            || normalize_projector_types(&pairing.compatible_vision_projector_types)
                                != normalized_base_projector_types
                    }
                    None => true,
                } {
                    entry = self.set_resolved_projector(
                        &entry.id,
                        &projector_id,
                        &base_plan.compatible_vision_projector_types,
                    )?;
                }
            } else {
                entry = self.detach_projector(&entry.id, base_plan)?;
            }
        }

        if base_plan.modality != ModelModality::Vision {
            return self.set_unresolved_pairing(
                &entry.id,
                base_plan,
                ModelPairingReason::BaseNotVision,
            );
        }
        if base_plan.compatible_vision_projector_types.is_empty() {
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
        }) {
            return Ok(entry);
        }

        let matches = self.find_compatible_installed_projector_ids(
            &base_plan.compatible_vision_projector_types,
            classified_projectors,
        );
        if matches.len() == 1 {
            self.set_resolved_projector(
                &entry.id,
                &matches[0],
                &base_plan.compatible_vision_projector_types,
            )
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

    fn find_compatible_installed_projector_ids(
        &self,
        compatible_vision_projector_types: &[String],
        classified_projectors: &[ClassifiedAsset],
    ) -> Vec<String> {
        let compatible = compatible_projector_type_set(compatible_vision_projector_types);
        let supplied: BTreeMap<_, _> = classified_projectors
            .iter()
            .map(|asset| (asset.asset_id.as_str(), &asset.inspection))
            .collect();
        let mut matches = Vec::new();
        for asset in self.manifest.assets.values() {
            if asset.kind != ModelAssetKind::Projector || asset.ref_count == 0 {
                continue;
            }
            let inspection = asset
                .inspection
                .as_ref()
                .or_else(|| supplied.get(asset.id.as_str()).copied());
            if projector_type_matches(inspection, &compatible) {
                matches.push(asset.id.clone());
            }
        }
        sorted_values(matches)
    }

    fn set_resolved_projector(
        &mut self,
        id: &str,
        projector_asset_id: &str,
        compatible_vision_projector_types: &[String],
    ) -> Result<BrowserModelEntry, ModelError> {
        let now = now_iso();
        let revision = self.manifest.projector_index_revision;
        self.update_model_entry(id, |entry| {
            entry.projector_asset_id = Some(projector_asset_id.to_string());
            entry.modality = ModelModality::Vision;
            entry.status = ModelStatus::Ready;
            entry.pairing = Some(browser_pairing(
                CoreModelPairingState::Resolved,
                revision,
                compatible_vision_projector_types,
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

    fn restore_entry(&mut self, snapshot: BrowserModelEntry) -> Result<(), ModelError> {
        let existing = self
            .manifest
            .models
            .get(&snapshot.id)
            .cloned()
            .ok_or_else(|| model_not_found(&snapshot.id))?;
        let previous_refs = entry_asset_ids(&existing);
        let next_refs = entry_asset_ids(&snapshot);
        self.rebalance_refs(&previous_refs, &next_refs)?;
        self.manifest.models.insert(snapshot.id.clone(), snapshot);
        validate_manifest(&self.manifest)
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
        let current = current_loaded_model(self.current.as_ref(), &entry.id);
        BrowserModelInfo {
            id: entry.id.clone(),
            name: entry.name.clone(),
            modality: entry.modality,
            status: entry.status,
            source: summary.source,
            bytes: summary.bytes,
            loaded: current.is_some(),
            chat_template: current.and_then(|current| current.chat_template.clone()),
            bos_text: current.map_or_else(String::new, |current| current.bos_text.clone()),
            eos_text: current.map_or_else(String::new, |current| current.eos_text.clone()),
            media_marker: browser_media_marker(current, entry.modality),
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
    let code = match error {
        ModelError::InvalidModelSource(_) => CODE_INVALID_MODEL_SOURCE,
        ModelError::InvalidModelPairing(_) => CODE_INVALID_MODEL_PAIRING,
        ModelError::StorageUnavailable(_) => CODE_STORAGE_UNAVAILABLE,
        ModelError::StorageCorrupt(_) => CODE_STORAGE_CORRUPT,
        ModelError::AssetMissing(_) => CODE_MODEL_BROKEN,
        ModelError::ModelNotFound(_) => CODE_MODEL_NOT_FOUND,
        ModelError::RemoteUnavailable(_) => CODE_REMOTE_LOAD_FAILED,
        ModelError::Runtime(_) => CODE_QUERY_FAILED,
        ModelError::RegistryJson(_) => CODE_STORAGE_CORRUPT,
        ModelError::Io(_) => CODE_STORAGE_UNAVAILABLE,
        ModelError::UnsupportedGgufVersion(_)
        | ModelError::InvalidGgufMetadata(_)
        | ModelError::GgufMetadataTooLarge { .. } => CODE_INVALID_MODEL_SOURCE,
    };
    BrowserLifecycleEnvelope {
        ok: false,
        value: None,
        error: Some(BrowserLifecycleError {
            code,
            message: error.to_string(),
        }),
    }
}

pub fn response_json<T>(response: BrowserLifecycleEnvelope<T>) -> String
where
    T: Serialize,
{
    serde_json::to_string(&response)
        .unwrap_or_else(|_| LIFECYCLE_SERIALIZATION_FALLBACK.to_string())
}

fn migrate_manifest(
    mut manifest: BrowserRegistryManifest,
) -> Result<BrowserRegistryManifest, ModelError> {
    validate_registry_manifest_version(BROWSER_REGISTRY_MANIFEST_LABEL, manifest.version)?;
    for (id, asset) in &mut manifest.assets {
        if asset.id.is_empty() {
            asset.id = id.clone();
        }
        if asset.created_at.trim().is_empty() {
            asset.created_at = now_iso();
        }
    }
    for (id, model) in &mut manifest.models {
        if model.id.is_empty() {
            model.id = id.clone();
        }
        if model.created_at.trim().is_empty() {
            model.created_at = now_iso();
        }
        if model.updated_at.trim().is_empty() {
            model.updated_at = model.created_at.clone();
        }
    }
    Ok(manifest)
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

fn resolve_source_projector_asset_id(
    classified: &[ClassifiedAsset],
    explicit_projector_asset_id: Option<&str>,
) -> Option<String> {
    if let Some(explicit) = explicit_projector_asset_id {
        return Some(explicit.to_string());
    }
    let projectors: Vec<_> = classified
        .iter()
        .filter(|asset| asset.inspection.role == crate::lifecycle::AssetRole::Projector)
        .collect();
    (projectors.len() == 1).then(|| projectors[0].asset_id.clone())
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

fn current_loaded_model<'model>(
    current: Option<&'model BrowserLoadedModelState>,
    entry_id: &str,
) -> Option<&'model BrowserLoadedModelState> {
    current.filter(|current| current.id == entry_id)
}

fn browser_media_marker(
    current: Option<&BrowserLoadedModelState>,
    modality: ModelModality,
) -> Option<String> {
    current
        .and_then(|current| current.media_marker.clone())
        .or_else(|| media_marker_for_modality(modality))
}

fn entry_asset_ids(entry: &BrowserModelEntry) -> Vec<String> {
    sorted_model_asset_ids(&entry.model_asset_ids, entry.projector_asset_id.as_ref())
}

fn failed_query_observation(message: Option<String>) -> BrowserQueryObservation {
    BrowserQueryObservation {
        session: None,
        status: QUERY_STATUS_FAILED.to_string(),
        wall_ms: None,
        ttft_ms: None,
        output_tokens: None,
        error_code: Some(CODE_QUERY_FAILED.to_string()),
        error_message: message,
    }
}

fn normalize_projector_types(projector_types: &[String]) -> Vec<String> {
    sorted_unique_strings(projector_types.to_vec())
}

fn compatible_projector_type_set(projector_types: &[String]) -> BTreeSet<&String> {
    projector_types.iter().collect()
}

fn projector_type_matches(
    inspection: Option<&AssetInspection>,
    compatible_projector_types: &BTreeSet<&String>,
) -> bool {
    inspection
        .and_then(|inspection| inspection.provided_vision_projector_type.as_ref())
        .is_some_and(|provided| compatible_projector_types.contains(provided))
}

fn browser_pairing(
    state: CoreModelPairingState,
    projector_index_revision: u64,
    compatible_vision_projector_types: &[String],
    reason_code: Option<ModelPairingReason>,
    updated_at: &str,
) -> BrowserModelPairing {
    BrowserModelPairing {
        state,
        checked_projector_index_revision: projector_index_revision,
        compatible_vision_projector_types: normalize_projector_types(
            compatible_vision_projector_types,
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

fn browser_load_required(
    current: Option<&BrowserLoadedModelState>,
    model_id: &str,
    asset_fingerprint: &str,
    runtime_fingerprint: &str,
) -> bool {
    match current {
        Some(current) => {
            current.id != model_id
                || current.asset_fingerprint != asset_fingerprint
                || current.runtime_fingerprint != runtime_fingerprint
        }
        None => true,
    }
}

fn runtime_fingerprint(runtime_config: &Value, mode: BrowserObservabilityMode) -> String {
    stable_json_hash(&json!({
        "observability": mode,
        "runtime": runtime_config,
    }))
}

fn runtime_config_with_observability(mut runtime: Value, mode: BrowserObservabilityMode) -> Value {
    if !runtime.is_object() {
        runtime = json!({});
    }
    let runtime_metrics = matches!(
        mode,
        BrowserObservabilityMode::Runtime | BrowserObservabilityMode::Profile
    );
    let backend_profiling = matches!(mode, BrowserObservabilityMode::Profile);
    let object = runtime.as_object_mut().expect("object set above");
    let observability = object.entry("observability").or_insert_with(|| json!({}));
    if !observability.is_object() {
        *observability = json!({});
    }
    let observability = observability.as_object_mut().expect("object set above");
    observability.insert("runtime_metrics".to_string(), Value::Bool(runtime_metrics));
    observability.insert(
        "backend_profiling".to_string(),
        Value::Bool(backend_profiling),
    );
    runtime
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
mod tests {
    mod browser_tests;
}
