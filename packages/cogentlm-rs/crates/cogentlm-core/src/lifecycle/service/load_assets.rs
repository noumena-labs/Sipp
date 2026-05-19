use std::path::PathBuf;

use crate::lifecycle::storage::StorageBackend;
use crate::lifecycle::{ModelEntry, ModelError};

use super::ModelService;

#[derive(Debug)]
pub(super) struct LoadAssetPaths {
    pub(super) model_path: PathBuf,
    pub(super) projector_path: Option<PathBuf>,
}

impl<B: StorageBackend> ModelService<B> {
    pub(super) fn resolve_load_asset_paths(
        &self,
        entry: &ModelEntry,
    ) -> Result<LoadAssetPaths, ModelError> {
        let model_asset = entry
            .model_asset_ids
            .first()
            .ok_or_else(|| ModelError::StorageCorrupt("model has no assets".to_string()))?;
        let model_record = self
            .registry
            .asset(model_asset)
            .ok_or_else(|| ModelError::StorageCorrupt(format!("missing asset {model_asset}")))?;
        let model_path = self.assets.resolve_asset_path(model_record)?;

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

        Ok(LoadAssetPaths {
            model_path,
            projector_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lifecycle::storage::now_unix_ms;
    use crate::lifecycle::{
        AssetRecord, AssetRole, AssetSource, ModelAssetKind, ModelEntry, ModelModality, ModelStatus,
    };
    use std::fs;

    struct TempDir {
        path: PathBuf,
    }

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "cogentlm-core-load-assets-{}-{}",
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

    fn asset_record(id: &str, storage_path: impl Into<PathBuf>) -> AssetRecord {
        AssetRecord {
            id: id.to_string(),
            kind: ModelAssetKind::Model,
            name: format!("{id}.gguf"),
            hash: id.to_string(),
            bytes: 1,
            storage_path: storage_path.into(),
            source: AssetSource::Local {
                path: PathBuf::from(format!("{id}.gguf")),
                modified_unix_ms: None,
            },
            ref_count: 1,
            created_at_unix_ms: now_unix_ms(),
            inspection: Some(crate::lifecycle::AssetInspection {
                version: 1,
                role: AssetRole::Model,
                architecture: None,
                vision_capable: false,
                compatible_vision_projector_types: Vec::new(),
                provided_vision_projector_type: None,
            }),
        }
    }

    fn model_entry(asset_ids: Vec<String>) -> ModelEntry {
        ModelEntry {
            id: "model".to_string(),
            name: "model".to_string(),
            modality: ModelModality::Text,
            status: ModelStatus::Ready,
            model_asset_ids: asset_ids,
            projector_asset_id: None,
            pairing: None,
            runtime_fingerprint: None,
            last_loaded_at_unix_ms: None,
            created_at_unix_ms: now_unix_ms(),
            updated_at_unix_ms: now_unix_ms(),
        }
    }

    #[test]
    fn resolve_load_asset_paths_rejects_missing_model_asset() {
        let root = TempDir::new("missing-load-asset");
        let service = ModelService::local(root.path.join("store")).expect("service");
        let entry = model_entry(vec!["missing".to_string()]);

        let error = service
            .resolve_load_asset_paths(&entry)
            .expect_err("missing asset");

        assert!(
            matches!(error, ModelError::StorageCorrupt(message) if message.contains("missing asset"))
        );
    }

    #[test]
    fn resolve_load_asset_paths_returns_storage_path() {
        let root = TempDir::new("load-asset-path");
        let mut service = ModelService::local(root.path.join("store")).expect("service");
        let record = asset_record("asset-a", PathBuf::from("assets/asset-a.gguf"));
        service.registry.upsert_asset(record).expect("asset");
        let entry = model_entry(vec!["asset-a".to_string()]);

        let paths = service
            .resolve_load_asset_paths(&entry)
            .expect("load asset paths");

        assert!(paths.model_path.ends_with("assets/asset-a.gguf"));
        assert!(paths.projector_path.is_none());
    }
}
