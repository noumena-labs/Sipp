//! Managed model storage, acquisition, and runtime loading.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use futures::lock::Mutex;

use crate::engine::SippEngine;
use crate::lifecycle::acquisition::RemoteAcquisitionIds;

use super::backend_policy::BackendPolicy;
use super::storage::{now_unix_ms, LocalStorageBackend, StorageBackend};
use super::util::{invalid_pairing, invalid_source, model_not_found};
use super::{
    AssetStore, ManagedModel, ModelEntry, ModelError, ModelLoadOptions, ModelRegistry, ModelStatus,
};

mod helpers;
mod load_assets;
mod source_resolution;

use helpers::runtime_fingerprint;

/// Persistent store for models managed by a client.
pub struct ModelStore {
    state: Arc<Mutex<ModelStoreState<LocalStorageBackend>>>,
}

struct ModelStoreState<B: StorageBackend> {
    registry: ModelRegistry<B>,
    assets: AssetStore<B>,
    acquisition_ids: RemoteAcquisitionIds,
    usage: BTreeMap<String, usize>,
}

impl ModelStore {
    pub(crate) fn local(root: impl Into<PathBuf>) -> Result<Self, ModelError> {
        let backend = LocalStorageBackend::new(root);
        let registry = ModelRegistry::open(backend.clone())?;
        let assets = AssetStore::new(backend);
        assets.recover_acquisition_journals(&registry.manifest)?;
        Ok(Self {
            state: Arc::new(Mutex::new(ModelStoreState {
                registry,
                assets,
                acquisition_ids: RemoteAcquisitionIds::default(),
                usage: BTreeMap::new(),
            })),
        })
    }

    /// Install model files from the host filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error when the files are invalid or cannot be persisted.
    pub async fn install_files<P, I>(&self, model_paths: I) -> Result<ManagedModel, ModelError>
    where
        P: Into<PathBuf>,
        I: IntoIterator<Item = P>,
    {
        self.install_files_with_optional_projector(model_paths, None)
            .await
    }

    /// Install model and projector files from the host filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error when the files are invalid, incompatible, or cannot be
    /// persisted.
    pub async fn install_files_with_projector<P, I>(
        &self,
        model_paths: I,
        projector_path: impl Into<PathBuf>,
    ) -> Result<ManagedModel, ModelError>
    where
        P: Into<PathBuf>,
        I: IntoIterator<Item = P>,
    {
        self.install_files_with_optional_projector(model_paths, Some(projector_path.into()))
            .await
    }

    async fn install_files_with_optional_projector<P, I>(
        &self,
        model_paths: I,
        projector_path: Option<PathBuf>,
    ) -> Result<ManagedModel, ModelError>
    where
        P: Into<PathBuf>,
        I: IntoIterator<Item = P>,
    {
        let paths = model_paths.into_iter().map(Into::into).collect();
        let mut state = self.state.lock().await;
        let model_id = state.install_files(paths, projector_path)?;
        state.model(&model_id)
    }

    /// Install model files from HTTP(S) URLs.
    ///
    /// # Errors
    ///
    /// Returns an error when acquisition, validation, or persistence fails.
    pub async fn install_urls<U, I>(&self, model_urls: I) -> Result<ManagedModel, ModelError>
    where
        U: Into<String>,
        I: IntoIterator<Item = U>,
    {
        self.install_urls_with_optional_projector(model_urls, None)
            .await
    }

    /// Install model and projector files from HTTP(S) URLs.
    ///
    /// # Errors
    ///
    /// Returns an error when acquisition, validation, pairing, or persistence
    /// fails.
    pub async fn install_urls_with_projector<U, I>(
        &self,
        model_urls: I,
        projector_url: impl Into<String>,
    ) -> Result<ManagedModel, ModelError>
    where
        U: Into<String>,
        I: IntoIterator<Item = U>,
    {
        self.install_urls_with_optional_projector(model_urls, Some(projector_url.into()))
            .await
    }

    async fn install_urls_with_optional_projector<U, I>(
        &self,
        model_urls: I,
        projector_url: Option<String>,
    ) -> Result<ManagedModel, ModelError>
    where
        U: Into<String>,
        I: IntoIterator<Item = U>,
    {
        let urls = model_urls.into_iter().map(Into::into).collect();
        let mut state = self.state.lock().await;
        let model_id = state.install_urls(urls, projector_url).await?;
        state.model(&model_id)
    }

    /// List models in the store.
    pub async fn list(&self) -> Vec<ManagedModel> {
        self.state.lock().await.models()
    }

    /// Remove a model and assets no other model references.
    ///
    /// # Errors
    ///
    /// Returns an error when the model is missing, in use, or cannot be removed.
    pub async fn remove(&self, model_id: &str) -> Result<(), ModelError> {
        self.state.lock().await.remove(model_id)
    }

    pub(crate) async fn load_engine(
        &self,
        model_id: &str,
        options: ModelLoadOptions,
    ) -> Result<SippEngine, ModelError> {
        self.state.lock().await.load_engine(model_id, options).await
    }

    pub(crate) async fn replace_usage(&self, previous: Option<&str>, next: Option<&str>) {
        self.state.lock().await.replace_usage(previous, next);
    }
}

impl<B: StorageBackend> ModelStoreState<B> {
    fn model(&self, model_id: &str) -> Result<ManagedModel, ModelError> {
        let entry = self
            .registry
            .manifest
            .models
            .get(model_id)
            .ok_or_else(|| model_not_found(model_id))?;
        Ok(self.model_from_entry(entry))
    }

    fn models(&self) -> Vec<ManagedModel> {
        self.registry
            .manifest
            .models
            .values()
            .map(|entry| self.model_from_entry(entry))
            .collect()
    }

    fn model_from_entry(&self, entry: &ModelEntry) -> ManagedModel {
        let bytes = entry
            .model_asset_ids
            .iter()
            .chain(entry.projector_asset_id.iter())
            .filter_map(|id| self.registry.manifest.assets.get(id))
            .map(|asset| asset.bytes)
            .sum();
        ManagedModel {
            id: entry.id.clone(),
            name: entry.name.clone(),
            bytes,
            modality: entry.modality,
            status: entry.status,
        }
    }

    fn remove(&mut self, model_id: &str) -> Result<(), ModelError> {
        if self.usage.get(model_id).copied().unwrap_or_default() > 0 {
            return Err(ModelError::ModelInUse(model_id.to_string()));
        }
        let removed = self.registry.remove_model(model_id)?;
        self.registry.save()?;
        for asset in removed.orphaned_assets {
            self.assets.delete_asset(&asset)?;
        }
        Ok(())
    }

    async fn load_engine(
        &mut self,
        model_id: &str,
        options: ModelLoadOptions,
    ) -> Result<SippEngine, ModelError> {
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
        let engine = SippEngine::load(&load_assets.model_path, backend_plan.config)
            .await
            .map_err(ModelError::from)?;
        self.registry.update_model(&entry.id, |model| {
            model.last_loaded_at_unix_ms = Some(now_unix_ms());
            model.runtime_fingerprint = Some(runtime_fingerprint);
        })?;
        self.registry.save()?;
        Ok(engine)
    }

    fn replace_usage(&mut self, previous: Option<&str>, next: Option<&str>) {
        if let Some(model_id) = previous {
            self.release(model_id);
        }
        if let Some(model_id) = next {
            *self.usage.entry(model_id.to_string()).or_default() += 1;
        }
    }

    fn release(&mut self, model_id: &str) {
        let Some(count) = self.usage.get_mut(model_id) else {
            return;
        };
        *count -= 1;
        if *count == 0 {
            self.usage.remove(model_id);
        }
    }
}

#[cfg(test)]
#[path = "../../tests/lifecycle/service_tests.rs"]
mod service_tests;
