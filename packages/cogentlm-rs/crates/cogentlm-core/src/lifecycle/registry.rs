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
        let model = next
            .models
            .get_mut(model_id)
            .ok_or_else(|| ModelError::ModelNotFound(model_id.to_string()))?;
        update(model);
        model.updated_at_unix_ms = now_unix_ms();
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
        asset.ref_count = asset.ref_count.saturating_add(1);
    }
    manifest.projector_index_revision = manifest.projector_index_revision.saturating_add(1);
    Ok(())
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
    manifest.projector_index_revision = manifest.projector_index_revision.saturating_add(1);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::{AssetSource, ModelAssetKind, ModelModality, ModelStatus, PairingPlan};
    use std::fs;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cogentlm-core-registry-{}-{}",
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

    fn asset(id: &str) -> AssetRecord {
        AssetRecord {
            id: id.to_string(),
            kind: ModelAssetKind::Model,
            name: format!("{id}.gguf"),
            hash: id.trim_start_matches("asset-").to_string(),
            bytes: 4,
            storage_path: PathBuf::from("assets").join(id),
            source: AssetSource::Local {
                path: PathBuf::from(format!("{id}.gguf")),
                modified_unix_ms: None,
            },
            ref_count: 0,
            created_at_unix_ms: now_unix_ms(),
            inspection: None,
        }
    }

    fn model(id: &str, asset_id: &str) -> ModelEntry {
        ModelEntry {
            id: id.to_string(),
            name: id.to_string(),
            modality: ModelModality::Text,
            status: ModelStatus::Ready,
            model_asset_ids: vec![asset_id.to_string()],
            projector_asset_id: None,
            pairing: None,
            runtime_fingerprint: None,
            created_at_unix_ms: now_unix_ms(),
            updated_at_unix_ms: now_unix_ms(),
            last_loaded_at_unix_ms: None,
        }
    }

    #[test]
    fn registry_persists_assets_and_models() {
        let root = TempDir::new("persist");
        let mut registry = ModelRegistry::local(root.path.clone()).expect("registry");
        registry.upsert_asset(asset("asset-a")).expect("asset");
        registry
            .insert_model(model("model-a", "asset-a"))
            .expect("model");
        registry.save().expect("save");

        let loaded = ModelRegistry::local(root.path.clone()).expect("reload");
        assert_eq!(loaded.manifest().assets["asset-a"].ref_count, 1);
        assert_eq!(loaded.manifest().models["model-a"].name, "model-a");
    }

    #[test]
    fn registry_removes_model_and_returns_orphaned_assets() {
        let root = TempDir::new("remove");
        let mut registry = ModelRegistry::local(root.path.clone()).expect("registry");
        registry.upsert_asset(asset("asset-a")).expect("asset");
        registry
            .insert_model(model("model-a", "asset-a"))
            .expect("model");

        let removed = registry.remove_model("model-a").expect("remove");

        assert_eq!(removed.model.id, "model-a");
        assert_eq!(removed.orphaned_assets.len(), 1);
        assert!(registry.manifest().assets.is_empty());
        assert!(registry.manifest().models.is_empty());
    }

    #[test]
    fn registry_reports_corrupt_manifest() {
        let root = TempDir::new("corrupt");
        fs::write(root.path.join("registry.json"), b"{\"version\":2}").expect("corrupt manifest");

        let error = ModelRegistry::local(root.path.clone()).expect_err("corrupt");
        assert!(matches!(error, ModelError::StorageCorrupt(_)));
    }

    #[test]
    fn model_entry_helper_uses_pairing_plan() {
        let plan = PairingPlan {
            model_asset_ids: vec!["asset-a".to_string()],
            projector_asset_id: None,
            name: "planned".to_string(),
            modality: ModelModality::Text,
            status: ModelStatus::Ready,
            compatible_vision_projector_types: Vec::new(),
        };

        let entry = model_entry_from_assets("model-a", "model-a", &plan);

        assert_eq!(entry.model_asset_ids, vec!["asset-a"]);
        assert_eq!(entry.status, ModelStatus::Ready);
    }
}
