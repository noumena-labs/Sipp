use std::fs;
use std::path::Path;

use crate::lifecycle::registry::model_entry_from_assets;
use crate::lifecycle::storage::{modified_unix_ms, StorageBackend};
use crate::lifecycle::{
    AssetRecord, AssetSource, ModelAsset, ModelAssetKind, ModelAssets, ModelError, ModelSource,
    PairingResolver,
};

use super::helpers::{
    classified_asset_from_record, hash_file, model_id_from_plan, pairing_state_from_plan, same_path,
};
use super::{ModelService, ResolvedSource};

impl<B: StorageBackend> ModelService<B> {
    pub(super) fn resolve_source(
        &mut self,
        source: ModelSource,
    ) -> Result<ResolvedSource, ModelError> {
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

                let mut classified = Vec::with_capacity(installed.len());
                classified.extend(installed.iter().map(classified_asset_from_record));
                let plan = if let Some(projector_id) = explicit_projector_id.as_deref() {
                    PairingResolver::resolve_explicit(&classified, projector_id)?
                } else {
                    PairingResolver::resolve(&classified)?
                };
                let entry_id = model_id_from_plan(&plan);
                let mut entry = model_entry_from_assets(&entry_id, &plan.name, &plan);
                entry.pairing = Some(pairing_state_from_plan(&plan));
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
            if self.assets.resolve_asset_path(record).is_ok()
                && hash_file(path).is_ok_and(|hash| hash == record.hash)
            {
                return Ok(Some(record.clone()));
            }
        }

        Ok(None)
    }
}
