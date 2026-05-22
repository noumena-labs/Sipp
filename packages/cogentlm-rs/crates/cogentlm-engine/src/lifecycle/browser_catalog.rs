//! Browser lifecycle catalog.
//!
//! This module owns the browser-facing model registry contract without taking a
//! dependency on OPFS, `File`, `fetch`, or WORKERFS. The browser host installs
//! assets and mounts files; Rust owns the persisted lifecycle decisions.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::lifecycle::{
    AssetInspection, ClassifiedAsset, ModelAssetKind, ModelError, ModelModality,
    ModelPairingReason, ModelPairingState as CoreModelPairingState, ModelSourceKind, ModelStatus,
    PairingPlan, PairingResolver,
};

const MANIFEST_VERSION: u32 = 3;
const DEFAULT_MEDIA_MARKER: &str = "<image>";

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
            version: MANIFEST_VERSION,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrowserObservabilityMode {
    Off,
    Runtime,
    Profile,
}

impl Default for BrowserObservabilityMode {
    fn default() -> Self {
        Self::Off
    }
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

#[derive(Debug, Clone)]
pub struct BrowserLifecycleService {
    manifest: BrowserRegistryManifest,
    current: Option<BrowserLoadedModelState>,
    pending: Option<PendingLoad>,
    snapshot: BrowserObservabilitySnapshot,
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

    pub fn manifest(&self) -> &BrowserRegistryManifest {
        &self.manifest
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

    pub fn snapshot(&self) -> BrowserObservabilitySnapshot {
        self.snapshot.clone()
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
                    .ok_or_else(|| ModelError::ModelNotFound(id.clone()))?;
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
                            let projector = plan.projector_asset_id.clone().ok_or_else(|| {
                                ModelError::InvalidModelPairing(
                                    "explicit projector did not produce a projector id".to_string(),
                                )
                            })?;
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
        let load_id = stable_hash(
            stable_json(&json!({
                "modelId": entry.id,
                "assetFingerprint": asset_fingerprint(&entry),
                "runtimeFingerprint": runtime_fingerprint,
                "nonce": now_iso(),
            }))
            .as_bytes(),
        );
        let load_required = match self.current.as_ref() {
            Some(current) => {
                current.id != entry.id
                    || current.asset_fingerprint != asset_fingerprint(&entry)
                    || current.runtime_fingerprint != runtime_fingerprint
            }
            None => true,
        };

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
        let events = self.drain_events();

        Ok(BrowserPrepareLoadResponse {
            load_id,
            model,
            runtime_fingerprint,
            runtime_config,
            load_required,
            assets,
            projector,
            manifest: self.manifest.clone(),
            snapshot: self.snapshot.clone(),
            events,
        })
    }

    pub fn commit_load(
        &mut self,
        request: BrowserCommitLoadRequest,
    ) -> Result<BrowserCommitLoadResponse, ModelError> {
        let pending = self.pending.take().ok_or_else(|| {
            ModelError::InvalidModelSource("no pending browser model load".to_string())
        })?;
        if pending.load_id != request.load_id
            || pending.model_id != request.model_id
            || pending.runtime_fingerprint != request.runtime_fingerprint
        {
            self.pending = Some(pending);
            return Err(ModelError::InvalidModelSource(
                "browser model load commit does not match the pending load".to_string(),
            ));
        }

        let loaded_at = now_iso();
        {
            let entry = self
                .manifest
                .models
                .get_mut(&request.model_id)
                .ok_or_else(|| ModelError::ModelNotFound(request.model_id.clone()))?;
            entry.last_loaded_at = Some(loaded_at.clone());
            entry.runtime_fingerprint = Some(request.runtime_fingerprint.clone());
            entry.updated_at = loaded_at;
        }
        validate_manifest(&self.manifest)?;
        let entry = self
            .manifest
            .models
            .get(&request.model_id)
            .ok_or_else(|| ModelError::ModelNotFound(request.model_id.clone()))?
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
        let events = self.drain_events();
        Ok(BrowserCommitLoadResponse {
            model,
            manifest: self.manifest.clone(),
            snapshot: self.snapshot.clone(),
            events,
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
                query: Some(Some(BrowserQueryObservation {
                    session: None,
                    status: "failed".to_string(),
                    wall_ms: None,
                    ttft_ms: None,
                    output_tokens: None,
                    error_code: Some("QUERY_FAILED".to_string()),
                    error_message: message,
                })),
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
            .ok_or_else(|| ModelError::ModelNotFound(model_id.to_string()))?;
        for asset_id in entry_asset_ids(&removed) {
            let Some(asset) = self.manifest.assets.get_mut(&asset_id) else {
                continue;
            };
            if asset.ref_count > 0 {
                asset.ref_count -= 1;
            }
        }
        let orphan_ids: Vec<_> = self
            .manifest
            .assets
            .iter()
            .filter_map(|(id, asset)| (asset.ref_count == 0).then_some(id.clone()))
            .collect();
        let mut orphaned_assets = Vec::with_capacity(orphan_ids.len());
        let mut changed_projector_index = false;
        for id in orphan_ids {
            if let Some(asset) = self.manifest.assets.remove(&id) {
                changed_projector_index |= asset.kind == ModelAssetKind::Projector;
                orphaned_assets.push(asset);
            }
        }
        if self
            .current
            .as_ref()
            .is_some_and(|current| current.id == model_id)
        {
            self.current = None;
        }
        if changed_projector_index {
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
        let events = self.drain_events();
        Ok(BrowserRemoveResponse {
            removed,
            orphaned_assets,
            manifest: self.manifest.clone(),
            snapshot: self.snapshot.clone(),
            events,
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
        let next_refs = sorted_unique(plan.model_asset_ids.clone());
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
        let mut classified = Vec::with_capacity(entry.model_asset_ids.len());
        for asset_id in &entry.model_asset_ids {
            let record = self.manifest.assets.get(asset_id).ok_or_else(|| {
                ModelError::StorageCorrupt(format!("model references missing asset {asset_id}"))
            })?;
            classified.push(classified_asset_from_record(record));
        }
        PairingResolver::resolve(&classified)
    }

    fn resolve_entry_for_loading(
        &mut self,
        mut entry: BrowserModelEntry,
        base_plan: &PairingPlan,
        classified_projectors: &[ClassifiedAsset],
    ) -> Result<BrowserModelEntry, ModelError> {
        if let Some(projector_id) = entry.projector_asset_id.clone() {
            let projector = self.manifest.assets.get(&projector_id).cloned();
            if projector.is_none() {
                entry = self.detach_projector(&entry.id, base_plan)?;
            } else if !base_plan.compatible_vision_projector_types.is_empty() {
                let projector = projector.expect("checked above");
                let inspection = projector.inspection.clone().or_else(|| {
                    classified_projectors
                        .iter()
                        .find(|asset| asset.asset_id == projector_id)
                        .map(|asset| asset.inspection.clone())
                });
                let provided_type = inspection
                    .as_ref()
                    .and_then(|inspection| inspection.provided_vision_projector_type.as_deref());
                if match provided_type {
                    Some(provided) => !base_plan
                        .compatible_vision_projector_types
                        .iter()
                        .any(|expected| expected == provided),
                    None => true,
                } {
                    entry = self.detach_projector(&entry.id, base_plan)?;
                } else if match entry.pairing.as_ref() {
                    Some(pairing) => {
                        pairing.state != CoreModelPairingState::Resolved
                            || normalize_projector_types(&pairing.compatible_vision_projector_types)
                                != normalize_projector_types(
                                    &base_plan.compatible_vision_projector_types,
                                )
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
                return Ok(entry);
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
                    == normalize_projector_types(&base_plan.compatible_vision_projector_types)
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
        let compatible: BTreeSet<_> = compatible_vision_projector_types.iter().collect();
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
            let provided = inspection
                .and_then(|inspection| inspection.provided_vision_projector_type.as_ref());
            if provided.map_or(false, |provided| compatible.contains(provided)) {
                matches.push(asset.id.clone());
            }
        }
        matches.sort();
        matches
    }

    fn set_resolved_projector(
        &mut self,
        id: &str,
        projector_asset_id: &str,
        compatible_vision_projector_types: &[String],
    ) -> Result<BrowserModelEntry, ModelError> {
        let now = now_iso();
        let mut entry = self
            .manifest
            .models
            .get(id)
            .cloned()
            .ok_or_else(|| ModelError::ModelNotFound(id.to_string()))?;
        let previous_refs = entry_asset_ids(&entry);
        entry.projector_asset_id = Some(projector_asset_id.to_string());
        entry.modality = ModelModality::Vision;
        entry.status = ModelStatus::Ready;
        entry.pairing = Some(BrowserModelPairing {
            state: CoreModelPairingState::Resolved,
            checked_projector_index_revision: self.manifest.projector_index_revision,
            compatible_vision_projector_types: normalize_projector_types(
                compatible_vision_projector_types,
            ),
            reason_code: None,
            updated_at: now.clone(),
        });
        entry.updated_at = now;
        let next_refs = entry_asset_ids(&entry);
        self.rebalance_refs(&previous_refs, &next_refs)?;
        self.manifest.models.insert(id.to_string(), entry.clone());
        validate_manifest(&self.manifest)?;
        Ok(entry)
    }

    fn set_unresolved_pairing(
        &mut self,
        id: &str,
        plan: &PairingPlan,
        reason_code: ModelPairingReason,
    ) -> Result<BrowserModelEntry, ModelError> {
        let now = now_iso();
        let mut entry = self
            .manifest
            .models
            .get(id)
            .cloned()
            .ok_or_else(|| ModelError::ModelNotFound(id.to_string()))?;
        let previous_refs = entry_asset_ids(&entry);
        entry.projector_asset_id = None;
        entry.modality = plan.modality;
        entry.status = plan.status;
        entry.pairing = Some(BrowserModelPairing {
            state: CoreModelPairingState::Unresolved,
            checked_projector_index_revision: self.manifest.projector_index_revision,
            compatible_vision_projector_types: normalize_projector_types(
                &plan.compatible_vision_projector_types,
            ),
            reason_code: Some(reason_code),
            updated_at: now.clone(),
        });
        entry.updated_at = now;
        let next_refs = entry_asset_ids(&entry);
        self.rebalance_refs(&previous_refs, &next_refs)?;
        self.manifest.models.insert(id.to_string(), entry.clone());
        validate_manifest(&self.manifest)?;
        Ok(entry)
    }

    fn detach_projector(
        &mut self,
        id: &str,
        base_plan: &PairingPlan,
    ) -> Result<BrowserModelEntry, ModelError> {
        let now = now_iso();
        let mut entry = self
            .manifest
            .models
            .get(id)
            .cloned()
            .ok_or_else(|| ModelError::ModelNotFound(id.to_string()))?;
        let previous_refs = entry_asset_ids(&entry);
        entry.projector_asset_id = None;
        entry.modality = base_plan.modality;
        entry.status = base_plan.status;
        entry.pairing = None;
        entry.updated_at = now;
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
            .ok_or_else(|| ModelError::ModelNotFound(snapshot.id.clone()))?;
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
        let mut assets = Vec::with_capacity(entry.model_asset_ids.len());
        for asset_id in &entry.model_asset_ids {
            let record = self.manifest.assets.get(asset_id).ok_or_else(|| {
                ModelError::StorageCorrupt(format!("model references missing asset {asset_id}"))
            })?;
            assets.push(planned_asset(record));
        }
        let projector = entry
            .projector_asset_id
            .as_deref()
            .map(|asset_id| {
                let record = self.manifest.assets.get(asset_id).ok_or_else(|| {
                    ModelError::StorageCorrupt(format!("model references missing asset {asset_id}"))
                })?;
                Ok::<BrowserPlannedAsset, ModelError>(planned_asset(record))
            })
            .transpose()?;
        Ok((assets, projector))
    }

    fn model_info_from_entry(&self, entry: &BrowserModelEntry) -> BrowserModelInfo {
        let mut bytes = 0_u64;
        let mut source = ModelSourceKind::Local;
        for asset_id in entry_asset_ids(entry) {
            if let Some(asset) = self.manifest.assets.get(&asset_id) {
                bytes = bytes.saturating_add(asset.bytes);
                if asset.source_url.is_some() {
                    source = ModelSourceKind::Remote;
                }
            }
        }
        let loaded = self
            .current
            .as_ref()
            .is_some_and(|current| current.id == entry.id);
        let current = loaded.then(|| self.current.as_ref()).flatten();
        BrowserModelInfo {
            id: entry.id.clone(),
            name: entry.name.clone(),
            modality: entry.modality,
            status: entry.status,
            source,
            bytes,
            loaded,
            chat_template: current.and_then(|current| current.chat_template.clone()),
            bos_text: current.map_or_else(String::new, |current| current.bos_text.clone()),
            eos_text: current.map_or_else(String::new, |current| current.eos_text.clone()),
            media_marker: current
                .and_then(|current| current.media_marker.clone())
                .or_else(|| {
                    (entry.modality == ModelModality::Vision)
                        .then(|| DEFAULT_MEDIA_MARKER.to_string())
                }),
            created_at: entry.created_at.clone(),
            updated_at: entry.updated_at.clone(),
        }
    }

    fn increment_refs(&mut self, asset_ids: &[String]) -> Result<(), ModelError> {
        for id in sorted_unique(asset_ids.to_vec()) {
            let asset = self.manifest.assets.get_mut(&id).ok_or_else(|| {
                ModelError::StorageCorrupt(format!("model references missing asset {id}"))
            })?;
            asset.ref_count = asset.ref_count.checked_add(1).ok_or_else(|| {
                ModelError::StorageCorrupt(format!("asset {id} refcount overflow"))
            })?;
        }
        Ok(())
    }

    fn decrement_refs(&mut self, asset_ids: &[String]) -> Result<(), ModelError> {
        for id in sorted_unique(asset_ids.to_vec()) {
            let asset = self.manifest.assets.get_mut(&id).ok_or_else(|| {
                ModelError::StorageCorrupt(format!("model references missing asset {id}"))
            })?;
            if asset.ref_count == 0 {
                return Err(ModelError::StorageCorrupt(format!(
                    "asset {id} refcount is already zero"
                )));
            }
            asset.ref_count -= 1;
        }
        Ok(())
    }

    fn rebalance_refs(
        &mut self,
        previous_refs: &[String],
        next_refs: &[String],
    ) -> Result<(), ModelError> {
        let previous: BTreeSet<_> = previous_refs.iter().cloned().collect();
        let next: BTreeSet<_> = next_refs.iter().cloned().collect();
        let removed: Vec<_> = previous.difference(&next).cloned().collect();
        let added: Vec<_> = next.difference(&previous).cloned().collect();
        self.decrement_refs(&removed)?;
        self.increment_refs(&added)
    }

    fn bump_projector_index_revision(&mut self) -> Result<(), ModelError> {
        self.manifest.projector_index_revision = self
            .manifest
            .projector_index_revision
            .checked_add(1)
            .ok_or_else(|| {
                ModelError::StorageCorrupt("projector index revision overflow".to_string())
            })?;
        Ok(())
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
        ModelError::InvalidModelSource(_) => "INVALID_MODEL_SOURCE",
        ModelError::InvalidModelPairing(_) => "INVALID_MODEL_PAIRING",
        ModelError::StorageUnavailable(_) => "STORAGE_UNAVAILABLE",
        ModelError::StorageCorrupt(_) => "STORAGE_CORRUPT",
        ModelError::AssetMissing(_) => "MODEL_BROKEN",
        ModelError::ModelNotFound(_) => "MODEL_NOT_FOUND",
        ModelError::RemoteUnavailable(_) => "REMOTE_LOAD_FAILED",
        ModelError::Runtime(_) => "QUERY_FAILED",
        ModelError::RegistryJson(_) => "STORAGE_CORRUPT",
        ModelError::Io(_) => "STORAGE_UNAVAILABLE",
        ModelError::UnsupportedGgufVersion(_)
        | ModelError::InvalidGgufMetadata(_)
        | ModelError::GgufMetadataTooLarge { .. } => "INVALID_MODEL_SOURCE",
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
    serde_json::to_string(&response).unwrap_or_else(|_| {
        "{\"ok\":false,\"error\":{\"code\":\"STORAGE_CORRUPT\",\"message\":\"failed to serialize lifecycle response\"}}".to_string()
    })
}

fn migrate_manifest(
    mut manifest: BrowserRegistryManifest,
) -> Result<BrowserRegistryManifest, ModelError> {
    if manifest.version != MANIFEST_VERSION {
        return Err(ModelError::StorageCorrupt(format!(
            "expected browser registry manifest version {MANIFEST_VERSION}, got {}",
            manifest.version
        )));
    }
    manifest.projector_index_revision = manifest.projector_index_revision.max(0);
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
    if manifest.version != MANIFEST_VERSION {
        return Err(ModelError::StorageCorrupt(format!(
            "expected browser registry manifest version {MANIFEST_VERSION}, got {}",
            manifest.version
        )));
    }
    let mut expected_ref_counts = BTreeMap::<String, u32>::new();
    for (id, asset) in &manifest.assets {
        if id != &asset.id {
            return Err(ModelError::StorageCorrupt(format!(
                "asset key {id} does not match record id {}",
                asset.id
            )));
        }
        validate_asset_record(asset)?;
    }
    for (id, model) in &manifest.models {
        if id != &model.id {
            return Err(ModelError::StorageCorrupt(format!(
                "model key {id} does not match record id {}",
                model.id
            )));
        }
        for asset_id in entry_asset_ids(model) {
            if !manifest.assets.contains_key(&asset_id) {
                return Err(ModelError::StorageCorrupt(format!(
                    "model {id} references missing asset {asset_id}"
                )));
            }
            let count = expected_ref_counts.entry(asset_id.clone()).or_default();
            *count = count.checked_add(1).ok_or_else(|| {
                ModelError::StorageCorrupt(format!("asset {asset_id} refcount overflow"))
            })?;
        }
    }
    for (id, asset) in &manifest.assets {
        let expected = expected_ref_counts.get(id).copied().unwrap_or(0);
        if asset.ref_count != expected {
            return Err(ModelError::StorageCorrupt(format!(
                "asset {id} refcount mismatch: stored {}, expected {expected}",
                asset.ref_count
            )));
        }
    }
    Ok(())
}

fn validate_asset_record(asset: &BrowserAssetRecord) -> Result<(), ModelError> {
    if asset.id.trim().is_empty() {
        return Err(ModelError::StorageCorrupt(
            "asset id must not be empty".to_string(),
        ));
    }
    if asset.storage_path.trim().is_empty() {
        return Err(ModelError::StorageCorrupt(format!(
            "asset {} storagePath must not be empty",
            asset.id
        )));
    }
    if asset.bytes == 0 {
        return Err(ModelError::StorageCorrupt(format!(
            "asset {} byte size must be positive",
            asset.id
        )));
    }
    let has_split_index = asset.source_part_index.is_some() || asset.source_part_count.is_some();
    if has_split_index {
        let index = asset.source_part_index.ok_or_else(|| {
            ModelError::StorageCorrupt(format!("asset {} split part index is missing", asset.id))
        })?;
        let count = asset.source_part_count.ok_or_else(|| {
            ModelError::StorageCorrupt(format!("asset {} split part count is missing", asset.id))
        })?;
        if count == 0 || index >= count {
            return Err(ModelError::StorageCorrupt(format!(
                "asset {} split part index/count is invalid",
                asset.id
            )));
        }
    }
    Ok(())
}

fn classified_asset_from_record(record: &BrowserAssetRecord) -> ClassifiedAsset {
    ClassifiedAsset {
        asset_id: record.id.clone(),
        name: record.name.clone(),
        inspection: record
            .inspection
            .clone()
            .unwrap_or_else(AssetInspection::unknown),
    }
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

fn entry_asset_ids(entry: &BrowserModelEntry) -> Vec<String> {
    let mut ids = entry.model_asset_ids.clone();
    if let Some(projector_id) = &entry.projector_asset_id {
        ids.push(projector_id.clone());
    }
    sorted_unique(ids)
}

fn sorted_unique(mut ids: Vec<String>) -> Vec<String> {
    ids.sort();
    ids.dedup();
    ids
}

fn normalize_projector_types(projector_types: &[String]) -> Vec<String> {
    sorted_unique(projector_types.to_vec())
}

fn base_model_id(plan: &PairingPlan) -> String {
    let hash = stable_hash(
        stable_json(&json!({
            "modelAssetIds": sorted_unique(plan.model_asset_ids.clone()),
        }))
        .as_bytes(),
    );
    format!("model-{}", &hash[..24])
}

fn asset_fingerprint(entry: &BrowserModelEntry) -> String {
    stable_hash(
        stable_json(&json!({
            "modelAssetIds": sorted_unique(entry.model_asset_ids.clone()),
            "projectorAssetId": entry.projector_asset_id,
        }))
        .as_bytes(),
    )
}

fn runtime_fingerprint(runtime_config: &Value, mode: BrowserObservabilityMode) -> String {
    stable_hash(
        stable_json(&json!({
            "observability": mode,
            "runtime": runtime_config,
        }))
        .as_bytes(),
    )
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

fn stable_hash(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
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
    let seconds = ms / 1000;
    let millis = ms % 1000;
    let days = (seconds / 86_400) as i64;
    let seconds_of_day = seconds % 86_400;
    let (year, month, day) = civil_from_days(days);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;
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
    use super::*;
    use crate::lifecycle::AssetRole;

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
            hash: id.to_string(),
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
        classified_asset_from_record(record)
    }

    #[test]
    fn prepares_and_commits_text_load() {
        let model = asset(
            "asset-model",
            ModelAssetKind::Model,
            inspection(AssetRole::Model, false, &[], None),
        );
        let mut service = BrowserLifecycleService::create(BrowserCreateConfig { manifest: None })
            .expect("service");

        let prepared = service
            .prepare_load(
                BrowserLoadSource::Assets {
                    assets: vec![model.clone()],
                    classified: vec![classified(&model)],
                    explicit_projector_asset_id: None,
                    classified_projectors: Vec::new(),
                },
                BrowserLoadOptions {
                    runtime: json!({ "context": { "n_ctx": 1024 } }),
                    observability: BrowserObservabilityMode::Runtime,
                },
            )
            .expect("prepare");

        assert!(prepared.load_required);
        assert_eq!(prepared.assets.len(), 1);
        assert_eq!(prepared.model.status, ModelStatus::Ready);
        assert_eq!(prepared.manifest.assets["asset-model"].ref_count, 1);

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
        let mut service = BrowserLifecycleService::create(BrowserCreateConfig { manifest: None })
            .expect("service");

        let first = service
            .prepare_load(
                BrowserLoadSource::Assets {
                    assets: vec![base.clone(), first_projector.clone()],
                    classified: vec![classified(&base), classified(&first_projector)],
                    explicit_projector_asset_id: Some(first_projector.id.clone()),
                    classified_projectors: Vec::new(),
                },
                BrowserLoadOptions {
                    runtime: json!({}),
                    observability: BrowserObservabilityMode::Off,
                },
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
                BrowserLoadOptions {
                    runtime: json!({}),
                    observability: BrowserObservabilityMode::Off,
                },
            )
            .expect_err("mismatched projector");

        assert!(matches!(error, ModelError::InvalidModelPairing(_)));
        let entry = service.manifest.models.get(&first.model.id).expect("entry");
        assert_eq!(entry.projector_asset_id.as_deref(), Some("asset-mmproj"));
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
        let asset = &service.manifest().assets["asset-a"];
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
}
