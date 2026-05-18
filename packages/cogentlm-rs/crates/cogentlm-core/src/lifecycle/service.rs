use std::fs;
use std::path::{Path, PathBuf};

use serde_json::json;
use sha2::{Digest, Sha256};

use crate::engine::{
    protocol::EngineStatus, ChatRequest, CogentEngine, EngineEventReceiver, EngineState,
    QueryRequest, RequestResult,
};

use super::backend_policy::{BackendPlan, BackendPolicy};
use super::registry::model_entry_from_assets;
use super::storage::{modified_unix_ms, now_unix_ms, LocalStorageBackend, StorageBackend};
use super::{
    AssetRecord, AssetSource, AssetStore, BackendSelection, ClassifiedAsset, ModelAsset,
    ModelAssetKind, ModelAssets, ModelError, ModelInfo, ModelLoadOptions, ModelModality,
    ModelPairing, ModelPairingReason, ModelPairingState, ModelRegistry, ModelServiceState,
    ModelSource, ModelSourceKind, ModelStatus, PairingPlan, PairingResolver,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedModelInfo {
    pub model: ModelInfo,
    pub backend: BackendSelection,
    pub runtime_fingerprint: String,
}

struct LoadedEngine {
    info: ModelInfo,
    runtime_fingerprint: String,
    engine: CogentEngine,
}

pub struct ModelService<B: StorageBackend = LocalStorageBackend> {
    registry: ModelRegistry<B>,
    assets: AssetStore<B>,
    current: Option<LoadedEngine>,
}

impl ModelService<LocalStorageBackend> {
    pub fn local(root: impl Into<PathBuf>) -> Result<Self, ModelError> {
        Self::open(LocalStorageBackend::new(root))
    }
}

impl<B: StorageBackend> ModelService<B> {
    pub fn open(backend: B) -> Result<Self, ModelError> {
        let registry = ModelRegistry::open(backend.clone())?;
        let assets = AssetStore::new(backend);
        Ok(Self {
            registry,
            assets,
            current: None,
        })
    }

    pub fn registry(&self) -> &ModelRegistry<B> {
        &self.registry
    }

    pub fn assets(&self) -> &AssetStore<B> {
        &self.assets
    }

    pub fn load(
        &mut self,
        source: ModelSource,
        options: ModelLoadOptions,
    ) -> Result<LoadedModelInfo, ModelError> {
        let resolved = self.resolve_source(source)?;
        self.load_entry(&resolved.entry_id, options)
    }

    pub fn load_installed(
        &mut self,
        model_id: impl AsRef<str>,
        options: ModelLoadOptions,
    ) -> Result<LoadedModelInfo, ModelError> {
        self.load_entry(model_id.as_ref(), options)
    }

    pub fn unload(&mut self) -> Result<(), ModelError> {
        if let Some(current) = self.current.take() {
            current.engine.close().map_err(ModelError::from)?;
        }
        Ok(())
    }

    pub fn remove(&mut self, model_id: impl AsRef<str>) -> Result<(), ModelError> {
        let model_id = model_id.as_ref();
        if self
            .current
            .as_ref()
            .is_some_and(|loaded| loaded.info.id == model_id)
        {
            self.unload()?;
        }
        let removed = self.registry.remove_model(model_id)?;
        for asset in &removed.orphaned_assets {
            self.assets.delete_asset(asset)?;
        }
        self.registry.save()?;
        Ok(())
    }

    pub fn list(&self) -> Vec<ModelInfo> {
        self.registry
            .models()
            .into_iter()
            .map(|entry| {
                let loaded = self
                    .current
                    .as_ref()
                    .is_some_and(|current| current.info.id == entry.id);
                self.model_info_from_entry(entry, loaded)
            })
            .collect()
    }

    pub fn current(&self) -> Option<ModelInfo> {
        self.current.as_ref().map(|loaded| loaded.info.clone())
    }

    pub fn query(&self, request: impl Into<QueryRequest>) -> Result<RequestResult, ModelError> {
        self.engine()?.query(request).map_err(ModelError::from)
    }

    pub fn chat(&self, request: impl Into<ChatRequest>) -> Result<RequestResult, ModelError> {
        self.engine()?.chat(request).map_err(ModelError::from)
    }

    pub fn state(&self) -> Result<ModelServiceState, ModelError> {
        let Some(current) = &self.current else {
            return Ok(ModelServiceState {
                status: EngineStatus::Idle,
                updated_at_unix_ms: now_unix_ms(),
                ..ModelServiceState::default()
            });
        };
        Ok(service_state_from_engine_state(
            current.engine.state().map_err(ModelError::from)?,
            current.info.clone(),
        ))
    }

    pub fn subscribe_events(&self) -> Result<EngineEventReceiver, ModelError> {
        Ok(self.engine()?.subscribe_events())
    }

    pub fn close(&mut self) -> Result<(), ModelError> {
        self.unload()
    }

    fn engine(&self) -> Result<&CogentEngine, ModelError> {
        self.current
            .as_ref()
            .map(|loaded| &loaded.engine)
            .ok_or_else(|| ModelError::ModelNotFound("no model is loaded".to_string()))
    }

    fn load_entry(
        &mut self,
        model_id: &str,
        options: ModelLoadOptions,
    ) -> Result<LoadedModelInfo, ModelError> {
        let entry = self
            .registry
            .model(model_id)
            .ok_or_else(|| ModelError::ModelNotFound(model_id.to_string()))?
            .clone();
        if entry.status != ModelStatus::Ready {
            return Err(ModelError::InvalidModelPairing(format!(
                "model {} is not ready; status is {:?}",
                entry.id, entry.status
            )));
        }

        let model_asset = entry
            .model_asset_ids
            .first()
            .ok_or_else(|| ModelError::StorageCorrupt("model has no assets".to_string()))?
            .clone();
        let model_record = self
            .registry
            .asset(&model_asset)
            .ok_or_else(|| ModelError::StorageCorrupt(format!("missing asset {model_asset}")))?
            .clone();
        let model_path = self.assets.resolve_asset_path(&model_record)?;
        let projector_path = entry
            .projector_asset_id
            .as_ref()
            .map(|asset_id| {
                let record = self.registry.asset(asset_id).ok_or_else(|| {
                    ModelError::StorageCorrupt(format!("missing projector asset {asset_id}"))
                })?;
                self.assets.resolve_asset_path(record)
            })
            .transpose()?;

        let mut backend_plan = BackendPolicy::select(&options)?;
        if let Some(path) = &projector_path {
            backend_plan.config.multimodal.projector_path = Some(path.display().to_string());
        }
        let runtime_fingerprint = runtime_fingerprint(&entry, &backend_plan);
        let info = self.model_info_from_entry(&entry, true);

        if self.current.as_ref().is_some_and(|current| {
            current.info.id == entry.id && current.runtime_fingerprint == runtime_fingerprint
        }) {
            return Ok(LoadedModelInfo {
                model: info,
                backend: backend_plan.selection,
                runtime_fingerprint,
            });
        }

        self.unload()?;
        let engine =
            CogentEngine::load(&model_path, backend_plan.config).map_err(ModelError::from)?;
        self.registry.update_model(&entry.id, |model| {
            model.last_loaded_at_unix_ms = Some(now_unix_ms());
            model.runtime_fingerprint = Some(runtime_fingerprint.clone());
        })?;
        self.registry.save()?;

        self.current = Some(LoadedEngine {
            info: info.clone(),
            runtime_fingerprint: runtime_fingerprint.clone(),
            engine,
        });

        Ok(LoadedModelInfo {
            model: info,
            backend: backend_plan.selection,
            runtime_fingerprint,
        })
    }

    fn resolve_source(&mut self, source: ModelSource) -> Result<ResolvedSource, ModelError> {
        match source {
            ModelSource::Installed { id } => {
                if self.registry.model(&id).is_none() {
                    return Err(ModelError::ModelNotFound(id));
                }
                Ok(ResolvedSource { entry_id: id })
            }
            ModelSource::Assets { model, projector } => {
                let mut installed = self.install_model_assets(model)?;
                let explicit_projector_id = if let Some(projector) = projector {
                    let projector = self.install_projector_asset(projector)?;
                    let id = projector.id.clone();
                    installed.push(projector);
                    Some(id)
                } else {
                    None
                };

                for record in &installed {
                    self.registry.upsert_asset(record.clone())?;
                }

                let classified = installed
                    .iter()
                    .map(classified_asset_from_record)
                    .collect::<Vec<_>>();
                let plan = if let Some(projector_id) = explicit_projector_id.as_deref() {
                    PairingResolver::resolve_explicit(&classified, projector_id)?
                } else {
                    PairingResolver::resolve(&classified)?
                };
                let entry_id = model_id_from_plan(&plan);
                let mut entry = model_entry_from_assets(&entry_id, &plan.name, &plan);
                entry.pairing = pairing_state_from_plan(&plan);
                self.registry.insert_model(entry)?;
                self.registry.save()?;
                Ok(ResolvedSource { entry_id })
            }
        }
    }

    fn install_model_assets(&self, assets: ModelAssets) -> Result<Vec<AssetRecord>, ModelError> {
        match assets {
            ModelAssets::Path { path } => self
                .install_local_asset(path, None)
                .map(|record| vec![record]),
            ModelAssets::Paths { paths } => {
                if paths.is_empty() {
                    return Err(ModelError::InvalidModelSource(
                        "model paths must not be empty".to_string(),
                    ));
                }
                paths
                    .into_iter()
                    .map(|path| self.install_local_asset(path, None))
                    .collect()
            }
            ModelAssets::Url { url } => Err(ModelError::RemoteUnavailable(url)),
            ModelAssets::Urls { urls } => Err(ModelError::RemoteUnavailable(urls.join(", "))),
        }
    }

    fn install_projector_asset(&self, asset: ModelAsset) -> Result<AssetRecord, ModelError> {
        match asset {
            ModelAsset::Path { path } => {
                self.install_local_asset(path, Some(ModelAssetKind::Projector))
            }
            ModelAsset::Url { url } => Err(ModelError::RemoteUnavailable(url)),
        }
    }

    fn install_local_asset(
        &self,
        path: impl AsRef<Path>,
        kind: Option<ModelAssetKind>,
    ) -> Result<AssetRecord, ModelError> {
        let path = path.as_ref();
        if let Some(record) = self.find_cached_local_asset(path, kind)? {
            return Ok(record);
        }

        self.assets
            .install_local_path_as(path, kind)
            .map(|installed| installed.record)
    }

    fn find_cached_local_asset(
        &self,
        path: &Path,
        kind: Option<ModelAssetKind>,
    ) -> Result<Option<AssetRecord>, ModelError> {
        let metadata = fs::metadata(path)?;
        if !metadata.is_file() {
            return Ok(None);
        }

        let source_path = fs::canonicalize(path)?;
        let source_modified_unix_ms = modified_unix_ms(&metadata);

        for record in self.registry.manifest().assets.values() {
            if kind.is_some_and(|expected| record.kind != expected) {
                continue;
            }
            if record.bytes != metadata.len() {
                continue;
            }

            let AssetSource::Local {
                path: record_source_path,
                modified_unix_ms: record_modified_unix_ms,
            } = &record.source
            else {
                continue;
            };

            if !same_path(record_source_path, &source_path) {
                continue;
            }
            if record_modified_unix_ms.is_some()
                && source_modified_unix_ms.is_some()
                && record_modified_unix_ms != &source_modified_unix_ms
            {
                continue;
            }
            if self.assets.resolve_asset_path(record).is_ok() {
                return Ok(Some(record.clone()));
            }
        }

        Ok(None)
    }

    fn model_info_from_entry(&self, entry: &super::ModelEntry, loaded: bool) -> ModelInfo {
        let mut bytes = 0_u64;
        let mut source = ModelSourceKind::Local;
        for asset_id in entry
            .model_asset_ids
            .iter()
            .chain(entry.projector_asset_id.iter())
        {
            if let Some(asset) = self.registry.asset(asset_id) {
                bytes = bytes.saturating_add(asset.bytes);
                if matches!(asset.source, AssetSource::Remote { .. }) {
                    source = ModelSourceKind::Remote;
                }
            }
        }

        ModelInfo {
            id: entry.id.clone(),
            name: entry.name.clone(),
            modality: entry.modality,
            status: entry.status,
            source,
            bytes,
            loaded,
            chat_template: None,
            bos_text: String::new(),
            eos_text: String::new(),
            media_marker: (entry.modality == ModelModality::Vision).then(|| "<image>".to_string()),
            created_at_unix_ms: entry.created_at_unix_ms,
            updated_at_unix_ms: entry.updated_at_unix_ms,
        }
    }
}

fn same_path(left: &Path, right: &Path) -> bool {
    #[cfg(windows)]
    {
        left.to_string_lossy()
            .eq_ignore_ascii_case(&right.to_string_lossy())
    }
    #[cfg(not(windows))]
    {
        left == right
    }
}

impl<B: StorageBackend> Drop for ModelService<B> {
    fn drop(&mut self) {
        let _ = self.unload();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedSource {
    entry_id: String,
}

fn classified_asset_from_record(record: &AssetRecord) -> ClassifiedAsset {
    ClassifiedAsset {
        asset_id: record.id.clone(),
        name: record.name.clone(),
        inspection: record
            .inspection
            .clone()
            .unwrap_or_else(super::AssetInspection::unknown),
    }
}

fn pairing_state_from_plan(plan: &PairingPlan) -> Option<ModelPairing> {
    let reason = match plan.status {
        ModelStatus::Ready => None,
        ModelStatus::NeedsProjector => Some(ModelPairingReason::NoMatch),
        ModelStatus::Broken => Some(ModelPairingReason::MissingMetadata),
    };
    Some(ModelPairing {
        state: if plan.status == ModelStatus::Ready {
            ModelPairingState::Resolved
        } else {
            ModelPairingState::Unresolved
        },
        checked_projector_index_revision: 0,
        compatible_vision_projector_types: plan.compatible_vision_projector_types.clone(),
        reason,
        updated_at_unix_ms: now_unix_ms(),
    })
}

fn model_id_from_plan(plan: &PairingPlan) -> String {
    let mut ids = plan.model_asset_ids.clone();
    if let Some(projector) = &plan.projector_asset_id {
        ids.push(projector.clone());
    }
    ids.sort();
    ids.dedup();
    format!("model-{}", stable_hash(ids.join("\n").as_bytes()))
}

fn runtime_fingerprint(entry: &super::ModelEntry, backend_plan: &BackendPlan) -> String {
    let mut model_asset_ids = entry.model_asset_ids.clone();
    model_asset_ids.sort();
    let value = json!({
        "modelAssetIds": model_asset_ids,
        "projectorAssetId": entry.projector_asset_id,
        "backend": backend_plan.selection.selected,
        "runtime": serde_json::to_value(&backend_plan.config)
            .unwrap_or_else(|_| json!(format!("{:?}", backend_plan.config))),
    });
    stable_hash(value.to_string().as_bytes())
}

fn stable_hash(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(digest.len() * 2);
    for byte in digest {
        output.push_str(&format!("{byte:02x}"));
    }
    output
}

fn service_state_from_engine_state(state: EngineState, model: ModelInfo) -> ModelServiceState {
    ModelServiceState {
        status: state.status,
        model: Some(model),
        backend: state.backend,
        runtime: state.runtime,
        requests: state.requests,
        stats: state.stats,
        updated_at_unix_ms: state.updated_at_unix_ms,
    }
}

pub fn model_source_from_path(path: impl AsRef<Path>) -> ModelSource {
    ModelSource::Assets {
        model: ModelAssets::Path {
            path: path.as_ref().to_path_buf(),
        },
        projector: None,
    }
}

pub fn vision_model_source_from_paths(
    model: impl AsRef<Path>,
    projector: impl AsRef<Path>,
) -> ModelSource {
    ModelSource::Assets {
        model: ModelAssets::Path {
            path: model.as_ref().to_path_buf(),
        },
        projector: Some(ModelAsset::Path {
            path: projector.as_ref().to_path_buf(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::{AssetInspection, AssetRole};
    use std::fs;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cogentlm-core-service-{}-{}",
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
}
