//! Managed model storage, acquisition, and runtime loading.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use futures::lock::Mutex;

use crate::engine::NativeRuntimeConfig;
#[cfg(not(target_family = "wasm"))]
use crate::lifecycle::acquisition::RemoteAcquisitionIds;

use super::backend_policy::BackendPolicy;
use super::storage::{modified_unix_ms, now_unix_ms, LocalStorageBackend, StorageBackend};
use super::util::{invalid_pairing, invalid_source, model_not_found};
use super::{
    AssetSource, AssetStore, ManagedModel, ModelEntry, ModelError, ModelLoadOptions,
    ModelRegistration, ModelRegistry, ModelStatus,
};

mod helpers;
mod load_assets;
mod source_resolution;

use helpers::runtime_fingerprint;

/// Persistent store for models managed by a client.
pub struct ModelStore {
    state: Arc<Mutex<ModelStoreState<LocalStorageBackend>>>,
}

/// Fully resolved native activation input that owns no runtime resources.
pub(crate) struct ModelActivationPlan {
    pub(crate) model_id: String,
    pub(crate) model_path: PathBuf,
    pub(crate) runtime: NativeRuntimeConfig,
    runtime_fingerprint: String,
}

struct ModelStoreState<B: StorageBackend> {
    registry: ModelRegistry<B>,
    assets: AssetStore<B>,
    #[cfg(not(target_family = "wasm"))]
    acquisition_ids: RemoteAcquisitionIds,
    usage: BTreeMap<String, usize>,
}

impl ModelStore {
    pub(crate) fn local(root: impl Into<PathBuf>) -> Result<Self, ModelError> {
        Self::local_with_backend(LocalStorageBackend::new(root))
    }

    pub(crate) fn local_with_source_root(
        root: impl Into<PathBuf>,
        local_source_root: impl Into<PathBuf>,
    ) -> Result<Self, ModelError> {
        Self::local_with_backend(LocalStorageBackend::with_local_source_root(
            root,
            local_source_root,
        ))
    }

    fn local_with_backend(backend: LocalStorageBackend) -> Result<Self, ModelError> {
        let registry = ModelRegistry::open(backend.clone())?;
        let assets = AssetStore::new(backend);
        assets.recover_acquisition_journals(&registry.manifest)?;
        let mut state = ModelStoreState {
            registry,
            assets,
            #[cfg(not(target_family = "wasm"))]
            acquisition_ids: RemoteAcquisitionIds::default(),
            usage: BTreeMap::new(),
        };
        state.prune_stale_local_models()?;
        Ok(Self {
            state: Arc::new(Mutex::new(state)),
        })
    }

    /// Add a model from local files or HTTP(S) URLs.
    ///
    /// # Errors
    ///
    /// Returns an error when sources are invalid, mix local and remote assets,
    /// or cannot be registered.
    pub async fn add<S, I>(&self, sources: I) -> Result<ManagedModel, ModelError>
    where
        S: AsRef<OsStr>,
        I: IntoIterator<Item = S>,
    {
        Ok(self.add_with_outcome(sources).await?.model)
    }

    /// Add a model and report whether the transaction created its model id.
    ///
    /// # Errors
    ///
    /// Returns an error when sources are invalid, mix local and remote assets,
    /// or cannot be registered.
    pub async fn add_with_outcome<S, I>(&self, sources: I) -> Result<ModelRegistration, ModelError>
    where
        S: AsRef<OsStr>,
        I: IntoIterator<Item = S>,
    {
        let sources = sources
            .into_iter()
            .map(|source| PathBuf::from(source.as_ref()))
            .collect();
        let mut state = self.state.lock().await;
        state.prune_stale_local_models()?;
        let outcome = state.add(sources).await?;
        Ok(ModelRegistration {
            model: state.model(&outcome.model_id)?,
            created: outcome.created,
        })
    }

    /// List models in the store.
    ///
    /// # Errors
    ///
    /// Returns an error when stale local registrations cannot be removed.
    pub async fn list(&self) -> Result<Vec<ManagedModel>, ModelError> {
        let mut state = self.state.lock().await;
        state.prune_stale_local_models()?;
        Ok(state.models())
    }

    /// Remove a model and assets no other model references.
    ///
    /// # Errors
    ///
    /// Returns an error when the model is missing, in use, or cannot be removed.
    pub async fn remove(&self, model_id: &str) -> Result<(), ModelError> {
        self.state.lock().await.remove(model_id)
    }

    pub(crate) async fn prepare_activation(
        &self,
        model_id: &str,
        options: ModelLoadOptions,
    ) -> Result<ModelActivationPlan, ModelError> {
        let (entry, load_assets) = self.state.lock().await.resolve_activation(model_id)?;
        let mut backend_plan = BackendPolicy::select(&options)?;
        if let Some(path) = &load_assets.projector_path {
            backend_plan.config.multimodal.projector_path = Some(path.display().to_string());
        }
        let runtime_fingerprint = runtime_fingerprint(&entry, &backend_plan)?;

        Ok(ModelActivationPlan {
            model_id: entry.id,
            model_path: load_assets.model_path,
            runtime: backend_plan.config,
            runtime_fingerprint,
        })
    }

    pub(crate) async fn commit_activation(
        &self,
        activation: &ModelActivationPlan,
    ) -> Result<(), ModelError> {
        self.state.lock().await.commit_activation(activation)
    }

    /// Records that an endpoint now holds `model_id`.
    pub(crate) async fn mark_used(&self, model_id: &str) {
        self.state.lock().await.mark_used(model_id);
    }

    /// Records that an endpoint no longer holds `model_id`.
    pub(crate) async fn mark_unused(&self, model_id: &str) {
        self.state.lock().await.mark_unused(model_id);
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
            self.assets.delete_managed_asset(&asset)?;
        }
        Ok(())
    }

    fn resolve_activation(
        &mut self,
        model_id: &str,
    ) -> Result<(ModelEntry, load_assets::LoadAssetPaths), ModelError> {
        self.prune_stale_local_models()?;
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
        Ok((entry, load_assets))
    }

    fn commit_activation(&mut self, activation: &ModelActivationPlan) -> Result<(), ModelError> {
        self.registry.update_model(&activation.model_id, |model| {
            model.last_loaded_at_unix_ms = Some(now_unix_ms());
            model.runtime_fingerprint = Some(activation.runtime_fingerprint.clone());
        })?;
        self.registry.save()?;
        Ok(())
    }

    fn prune_stale_local_models(&mut self) -> Result<(), ModelError> {
        let mut stale = Vec::new();
        for entry in self.registry.manifest.models.values() {
            if self.model_has_stale_local_asset(entry)? {
                stale.push(entry.id.clone());
            }
        }
        if stale.is_empty() {
            return Ok(());
        }

        let mut orphaned = Vec::new();
        for model_id in stale {
            orphaned.extend(self.registry.remove_model(&model_id)?.orphaned_assets);
        }
        self.registry.save()?;
        for asset in orphaned {
            self.assets.delete_managed_asset(&asset)?;
        }
        Ok(())
    }

    fn model_has_stale_local_asset(&self, entry: &ModelEntry) -> Result<bool, ModelError> {
        for asset_id in entry
            .model_asset_ids
            .iter()
            .chain(entry.projector_asset_id.iter())
        {
            let record = self
                .registry
                .manifest
                .assets
                .get(asset_id)
                .ok_or_else(|| ModelError::AssetMissing(asset_id.clone()))?;
            let AssetSource::Local {
                modified_unix_ms: expected_modified,
                ..
            } = &record.source
            else {
                continue;
            };
            if self.assets.requires_external_access(record) {
                continue;
            }
            let path = match self.assets.resolve_asset_path(record) {
                Ok(path) => path,
                Err(ModelError::AssetMissing(_)) => return Ok(true),
                Err(error) => return Err(error),
            };
            let metadata = match fs::metadata(path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(true),
                Err(error) => return Err(ModelError::Io(error)),
            };
            if !metadata.is_file() || metadata.len() != record.bytes {
                return Ok(true);
            }
            if expected_modified.is_some() && modified_unix_ms(&metadata) != *expected_modified {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn mark_used(&mut self, model_id: &str) {
        *self.usage.entry(model_id.to_string()).or_default() += 1;
    }

    fn mark_unused(&mut self, model_id: &str) {
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
