//! High-level lifecycle service: ingest sources, resolve pairings, expose ready models.

use std::path::PathBuf;

use crate::engine::SippEngine;
use crate::lifecycle::acquisition::RemoteAcquisitionIds;

use super::backend_policy::BackendPolicy;
use super::storage::{now_unix_ms, LocalStorageBackend, StorageBackend};
use super::util::{invalid_pairing, invalid_source, model_not_found};
use super::{AssetStore, ModelError, ModelLoadOptions, ModelRegistry, ModelSource, ModelStatus};

mod helpers;
mod load_assets;
mod source_resolution;

use helpers::runtime_fingerprint;

const NO_MODEL_LOADED: &str = "no model is loaded";

struct LoadedEngine {
    model_id: String,
    runtime_fingerprint: String,
    engine: SippEngine,
}

pub(crate) struct ModelService<B: StorageBackend = LocalStorageBackend> {
    registry: ModelRegistry<B>,
    assets: AssetStore<B>,
    current: Option<LoadedEngine>,
    acquisition_ids: RemoteAcquisitionIds,
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
        assets.recover_acquisition_journals(&registry.manifest)?;
        Ok(Self {
            registry,
            assets,
            current: None,
            acquisition_ids: RemoteAcquisitionIds::default(),
        })
    }

    pub async fn load(
        &mut self,
        source: ModelSource,
        options: ModelLoadOptions,
    ) -> Result<(), ModelError> {
        let resolved = self.resolve_source(source).await?;
        self.load_entry(&resolved.entry_id, options).await
    }

    pub async fn unload(&mut self) -> Result<(), ModelError> {
        if let Some(current) = self.current.take() {
            current.engine.close().await.map_err(ModelError::from)?;
        }
        Ok(())
    }

    pub(crate) fn take_loaded_engine(&mut self) -> Result<SippEngine, ModelError> {
        self.current
            .take()
            .map(|loaded| loaded.engine)
            .ok_or_else(|| model_not_found(NO_MODEL_LOADED))
    }

    async fn load_entry(
        &mut self,
        model_id: &str,
        options: ModelLoadOptions,
    ) -> Result<(), ModelError> {
        let entry = self
            .registry
            .manifest
            .models
            .get(model_id)
            .ok_or_else(|| model_not_found(model_id))?
            .clone();
        if entry.status != ModelStatus::Ready {
            return Err(invalid_pairing(format!(
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
        if self.is_loaded_model_with_fingerprint(&entry.id, &runtime_fingerprint) {
            return Ok(());
        }

        self.unload().await?;
        let engine = SippEngine::load(&load_assets.model_path, backend_plan.config)
            .await
            .map_err(ModelError::from)?;
        self.registry.update_model(&entry.id, |model| {
            model.last_loaded_at_unix_ms = Some(now_unix_ms());
            model.runtime_fingerprint = Some(runtime_fingerprint.clone());
        })?;
        self.registry.save()?;

        self.current = Some(LoadedEngine {
            model_id: entry.id,
            runtime_fingerprint,
            engine,
        });

        Ok(())
    }

    fn is_loaded_model_with_fingerprint(&self, model_id: &str, runtime_fingerprint: &str) -> bool {
        self.current.as_ref().is_some_and(|current| {
            current.model_id == model_id && current.runtime_fingerprint == runtime_fingerprint
        })
    }
}

impl<B: StorageBackend> Drop for ModelService<B> {
    fn drop(&mut self) {
        self.current.take();
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedSource {
    entry_id: String,
}

#[cfg(test)]
#[path = "../../tests/lifecycle/service_tests.rs"]
mod service_tests;
