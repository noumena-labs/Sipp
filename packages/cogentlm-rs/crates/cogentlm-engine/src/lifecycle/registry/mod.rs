//! On-disk model registry: read/insert/remove models and their referenced assets.

use std::path::PathBuf;

use crate::collection::remove_matching_values;

use super::storage::{now_unix_ms, LocalStorageBackend, StorageBackend};
use super::util::{empty_asset_id, model_not_found, storage_corrupt};
use super::{AssetRecord, ModelEntry, ModelError, RegistryManifest};

mod refs;

use refs::{
    decrement_refs, increment_refs, rebalance_refs, referenced_asset_ids, validate_manifest,
};

fn manifest_parse_failed(path: &std::path::Path, error: serde_json::Error) -> ModelError {
    storage_corrupt(format!("failed to parse {}: {}", path.display(), error))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemovedModel {
    pub model: ModelEntry,
    pub orphaned_assets: Vec<AssetRecord>,
}

#[derive(Debug, Clone)]
pub struct ModelRegistry<B = LocalStorageBackend> {
    backend: B,
    manifest: RegistryManifest,
}

impl ModelRegistry<LocalStorageBackend> {
    pub fn local(root: impl Into<PathBuf>) -> Result<Self, ModelError> {
        Self::open(LocalStorageBackend::new(root))
    }
}

impl<B: StorageBackend> ModelRegistry<B> {
    pub fn open(backend: B) -> Result<Self, ModelError> {
        backend.ensure_layout()?;
        let manifest_path = backend.manifest_path();
        let manifest = if manifest_path.exists() {
            let bytes = std::fs::read(&manifest_path)?;
            let manifest = serde_json::from_slice::<RegistryManifest>(&bytes)
                .map_err(|error| manifest_parse_failed(&manifest_path, error))?;
            validate_manifest(&manifest)?;
            manifest
        } else {
            RegistryManifest::default()
        };

        let registry = Self { backend, manifest };
        if !manifest_path.exists() {
            registry.save()?;
        }
        Ok(registry)
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn manifest(&self) -> &RegistryManifest {
        &self.manifest
    }

    pub fn save(&self) -> Result<(), ModelError> {
        validate_manifest(&self.manifest)?;
        let bytes = serde_json::to_vec_pretty(&self.manifest)?;
        self.backend
            .atomic_write(&self.backend.manifest_path(), &bytes)
    }

    pub fn upsert_asset(&mut self, mut record: AssetRecord) -> Result<(), ModelError> {
        if record.id.trim().is_empty() {
            return Err(empty_asset_id());
        }
        if let Some(existing) = self.manifest.assets.get(&record.id) {
            record.ref_count = existing.ref_count;
            record.created_at_unix_ms = existing.created_at_unix_ms;
        }
        self.manifest.assets.insert(record.id.clone(), record);
        Ok(())
    }

    pub fn insert_model(&mut self, entry: ModelEntry) -> Result<(), ModelError> {
        let mut next = self.manifest.clone();
        if let Some(existing) = next.models.remove(&entry.id) {
            decrement_refs(&mut next, referenced_asset_ids(&existing))?;
        }
        increment_refs(&mut next, referenced_asset_ids(&entry))?;
        next.models.insert(entry.id.clone(), entry);
        validate_manifest(&next)?;
        self.manifest = next;
        Ok(())
    }

    pub fn update_model(
        &mut self,
        model_id: &str,
        update: impl FnOnce(&mut ModelEntry),
    ) -> Result<(), ModelError> {
        let mut next = self.manifest.clone();
        let previous_refs = {
            let model = next
                .models
                .get(model_id)
                .ok_or_else(|| model_not_found(model_id))?;
            referenced_asset_ids(model)
        };
        let updated_refs = {
            let model = next
                .models
                .get_mut(model_id)
                .ok_or_else(|| model_not_found(model_id))?;
            update(model);
            model.updated_at_unix_ms = now_unix_ms();
            referenced_asset_ids(model)
        };

        rebalance_refs(&mut next, &previous_refs, &updated_refs)?;
        validate_manifest(&next)?;
        self.manifest = next;
        Ok(())
    }

    pub fn remove_model(&mut self, model_id: &str) -> Result<RemovedModel, ModelError> {
        let mut next = self.manifest.clone();
        let model = next
            .models
            .remove(model_id)
            .ok_or_else(|| model_not_found(model_id))?;
        decrement_refs(&mut next, referenced_asset_ids(&model))?;
        let orphaned_assets = remove_orphaned_assets(&mut next);

        validate_manifest(&next)?;
        self.manifest = next;
        Ok(RemovedModel {
            model,
            orphaned_assets,
        })
    }

    pub fn model(&self, model_id: &str) -> Option<&ModelEntry> {
        self.manifest.models.get(model_id)
    }

    pub fn asset(&self, asset_id: &str) -> Option<&AssetRecord> {
        self.manifest.assets.get(asset_id)
    }

    pub fn models(&self) -> Vec<&ModelEntry> {
        self.manifest.models.values().collect()
    }
}

fn remove_orphaned_assets(manifest: &mut RegistryManifest) -> Vec<AssetRecord> {
    remove_matching_values(&mut manifest.assets, |record| record.ref_count == 0)
}

pub fn model_entry_from_assets(
    id: impl Into<String>,
    name: impl Into<String>,
    plan: &super::PairingPlan,
) -> ModelEntry {
    let now = now_unix_ms();
    ModelEntry {
        id: id.into(),
        name: name.into(),
        modality: plan.modality,
        status: plan.status,
        model_asset_ids: plan.model_asset_ids.clone(),
        projector_asset_id: plan.projector_asset_id.clone(),
        pairing: None,
        runtime_fingerprint: None,
        created_at_unix_ms: now,
        updated_at_unix_ms: now,
        last_loaded_at_unix_ms: None,
    }
}

#[cfg(test)]
mod tests {
    mod registry_tests;
}
