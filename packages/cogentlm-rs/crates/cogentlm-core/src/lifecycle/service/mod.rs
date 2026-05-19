//! High-level lifecycle service: ingest sources, resolve pairings, expose ready models.

use std::path::{Path, PathBuf};

use crate::engine::{
    protocol::EngineStatus, ChatRequest, CogentEngine, EngineEventReceiver, QueryRequest,
    RequestResult,
};

use super::backend_policy::BackendPolicy;
use super::storage::{now_unix_ms, LocalStorageBackend, StorageBackend};
use super::{
    AssetSource, AssetStore, BackendSelection, ModelAsset, ModelAssets, ModelError, ModelInfo,
    ModelLoadOptions, ModelModality, ModelRegistry, ModelServiceState, ModelSource,
    ModelSourceKind, ModelStatus,
};

mod helpers;
mod load_assets;
mod source_resolution;

use helpers::{runtime_fingerprint, service_state_from_engine_state};

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

        let load_assets = self.resolve_load_asset_paths(&entry)?;

        let mut backend_plan = BackendPolicy::select(&options)?;
        if let Some(path) = &load_assets.projector_path {
            backend_plan.config.multimodal.projector_path = Some(path.display().to_string());
        }
        let runtime_fingerprint = runtime_fingerprint(&entry, &backend_plan)?;
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
        let engine = CogentEngine::load(&load_assets.model_path, backend_plan.config)
            .map_err(ModelError::from)?;
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

    fn model_info_from_entry(&self, entry: &super::ModelEntry, loaded: bool) -> ModelInfo {
        let mut bytes = 0_u64;
        let mut source = ModelSourceKind::Local;
        for asset_id in entry
            .model_asset_ids
            .iter()
            .chain(entry.projector_asset_id.iter())
        {
            if let Some(asset) = self.registry.asset(asset_id) {
                debug_assert!(bytes.checked_add(asset.bytes).is_some());
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

impl<B: StorageBackend> Drop for ModelService<B> {
    fn drop(&mut self) {
        let _ = self.unload();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedSource {
    entry_id: String,
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
    mod service_tests;
}
