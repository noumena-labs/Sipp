//! On-disk model registry: read/insert/remove models and their referenced assets.

use std::collections::BTreeMap;
use std::path::PathBuf;

use super::storage::{now_unix_ms, LocalStorageBackend, StorageBackend};
use super::{AssetRecord, ModelEntry, ModelError, RegistryManifest};

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
            let manifest = serde_json::from_slice::<RegistryManifest>(&bytes).map_err(|error| {
                ModelError::StorageCorrupt(format!(
                    "failed to parse {}: {}",
                    manifest_path.display(),
                    error
                ))
            })?;
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
            return Err(ModelError::StorageCorrupt(
                "asset id must not be empty".to_string(),
            ));
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
                .ok_or_else(|| ModelError::ModelNotFound(model_id.to_string()))?;
            referenced_asset_ids(model)
        };
        let updated_refs = {
            let model = next
                .models
                .get_mut(model_id)
                .ok_or_else(|| ModelError::ModelNotFound(model_id.to_string()))?;
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
            .ok_or_else(|| ModelError::ModelNotFound(model_id.to_string()))?;
        decrement_refs(&mut next, referenced_asset_ids(&model))?;

        let orphaned_ids: Vec<_> = next
            .assets
            .iter()
            .filter_map(|(id, record)| (record.ref_count == 0).then_some(id.clone()))
            .collect();
        let mut orphaned_assets = Vec::with_capacity(orphaned_ids.len());
        for id in orphaned_ids {
            if let Some(record) = next.assets.remove(&id) {
                orphaned_assets.push(record);
            }
        }

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

fn validate_manifest(manifest: &RegistryManifest) -> Result<(), ModelError> {
    if manifest.version != 3 {
        return Err(ModelError::StorageCorrupt(format!(
            "expected manifest version 3, got {}",
            manifest.version
        )));
    }
    let mut expected_ref_counts = BTreeMap::<String, u32>::new();
    for (id, asset) in &manifest.assets {
        if id != &asset.id {
            return Err(ModelError::StorageCorrupt(format!(
                "asset key {} does not match record id {}",
                id, asset.id
            )));
        }
    }
    for (id, model) in &manifest.models {
        if id != &model.id {
            return Err(ModelError::StorageCorrupt(format!(
                "model key {} does not match record id {}",
                id, model.id
            )));
        }
        for asset_id in referenced_asset_ids(model) {
            if !manifest.assets.contains_key(&asset_id) {
                return Err(ModelError::StorageCorrupt(format!(
                    "model {} references missing asset {}",
                    id, asset_id
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
                "asset {} refcount mismatch: stored {}, expected {}",
                id, asset.ref_count, expected
            )));
        }
    }
    Ok(())
}

fn referenced_asset_ids(model: &ModelEntry) -> Vec<String> {
    let mut ids = model.model_asset_ids.clone();
    if let Some(projector_id) = &model.projector_asset_id {
        ids.push(projector_id.clone());
    }
    ids.sort();
    ids.dedup();
    ids
}

fn increment_refs(
    manifest: &mut RegistryManifest,
    asset_ids: Vec<String>,
) -> Result<(), ModelError> {
    for id in asset_ids {
        let asset = manifest.assets.get_mut(&id).ok_or_else(|| {
            ModelError::StorageCorrupt(format!("model references missing asset {}", id))
        })?;
        asset.ref_count = asset
            .ref_count
            .checked_add(1)
            .ok_or_else(|| ModelError::StorageCorrupt(format!("asset {} refcount overflow", id)))?;
    }
    bump_projector_index_revision(manifest)?;
    Ok(())
}

fn rebalance_refs(
    manifest: &mut RegistryManifest,
    previous_refs: &[String],
    updated_refs: &[String],
) -> Result<(), ModelError> {
    let (removed_refs, added_refs) = sorted_ref_deltas(previous_refs, updated_refs);

    if removed_refs.is_empty() && added_refs.is_empty() {
        return Ok(());
    }

    decrement_refs(manifest, removed_refs)?;
    increment_refs(manifest, added_refs)
}

fn sorted_ref_deltas(
    previous_refs: &[String],
    updated_refs: &[String],
) -> (Vec<String>, Vec<String>) {
    let mut removed_refs = Vec::with_capacity(previous_refs.len());
    let mut added_refs = Vec::with_capacity(updated_refs.len());
    let mut previous = previous_refs.iter();
    let mut updated = updated_refs.iter();
    let mut previous_id = previous.next();
    let mut updated_id = updated.next();

    loop {
        match (previous_id, updated_id) {
            (Some(previous_value), Some(updated_value)) => {
                match previous_value.cmp(updated_value) {
                    std::cmp::Ordering::Less => {
                        removed_refs.push(previous_value.clone());
                        previous_id = previous.next();
                    }
                    std::cmp::Ordering::Equal => {
                        previous_id = previous.next();
                        updated_id = updated.next();
                    }
                    std::cmp::Ordering::Greater => {
                        added_refs.push(updated_value.clone());
                        updated_id = updated.next();
                    }
                }
            }
            (Some(previous_value), None) => {
                removed_refs.push(previous_value.clone());
                removed_refs.extend(previous.cloned());
                break;
            }
            (None, Some(updated_value)) => {
                added_refs.push(updated_value.clone());
                added_refs.extend(updated.cloned());
                break;
            }
            (None, None) => break,
        }
    }

    (removed_refs, added_refs)
}

fn decrement_refs(
    manifest: &mut RegistryManifest,
    asset_ids: Vec<String>,
) -> Result<(), ModelError> {
    for id in asset_ids {
        let asset = manifest.assets.get_mut(&id).ok_or_else(|| {
            ModelError::StorageCorrupt(format!("model references missing asset {}", id))
        })?;
        if asset.ref_count == 0 {
            return Err(ModelError::StorageCorrupt(format!(
                "asset {} refcount is already zero",
                id
            )));
        }
        asset.ref_count -= 1;
    }
    bump_projector_index_revision(manifest)?;
    Ok(())
}

fn bump_projector_index_revision(manifest: &mut RegistryManifest) -> Result<(), ModelError> {
    manifest.projector_index_revision = manifest
        .projector_index_revision
        .checked_add(1)
        .ok_or_else(|| {
            ModelError::StorageCorrupt("projector index revision overflow".to_string())
        })?;
    Ok(())
}

#[cfg(test)]
mod tests;
