use std::fs;
use std::path::Path;

use crate::lifecycle::registry::model_entry_from_assets;
use crate::lifecycle::storage::{hash_file, modified_unix_ms, now_unix_ms, StorageBackend};
use crate::lifecycle::util::classified_asset;
use crate::lifecycle::{
    AssetRecord, AssetSource, ModelAsset, ModelAssetKind, ModelAssets, ModelError, ModelPairing,
    ModelPairingReason, ModelPairingState, ModelSource, ModelStatus, PairingResolver,
};

use super::helpers::{model_id_from_plan, same_path};
use super::{invalid_source, model_not_found, ModelService, ResolvedSource};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/service/source_resolution_tests.rs"]
mod source_resolution_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const MODEL_PATHS_REQUIRED: &str = "model paths must not be empty";

impl<B: StorageBackend> ModelService<B> {
    pub(super) fn resolve_source(
        &mut self,
        source: ModelSource,
    ) -> Result<ResolvedSource, ModelError> {
        match source {
            ModelSource::Installed { id } => {
                if !self.registry.manifest.models.contains_key(&id) {
                    return Err(model_not_found(&id));
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

                self.register_installed_assets(&installed, explicit_projector_id.as_deref())
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
                    return Err(invalid_source(MODEL_PATHS_REQUIRED));
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

        for record in self.registry.manifest.assets.values() {
            if cached_local_record_matches(
                record,
                kind,
                metadata.len(),
                &source_path,
                source_modified_unix_ms,
            ) && self.assets.resolve_asset_path(record).is_ok()
                && hash_file(path).is_ok_and(|hash| hash == record.hash)
            {
                return Ok(Some(record.clone()));
            }
        }

        Ok(None)
    }

    fn register_installed_assets(
        &mut self,
        installed: &[AssetRecord],
        explicit_projector_id: Option<&str>,
    ) -> Result<ResolvedSource, ModelError> {
        let classified: Vec<_> = installed
            .iter()
            .map(|record| {
                classified_asset(
                    record.id.clone(),
                    record.name.clone(),
                    record.inspection.clone(),
                )
            })
            .collect();
        let plan = if let Some(projector_id) = explicit_projector_id {
            PairingResolver::resolve_explicit(&classified, projector_id)?
        } else {
            PairingResolver::resolve(&classified)?
        };
        let entry_id = model_id_from_plan(&plan);
        let mut entry = model_entry_from_assets(&entry_id, &plan.name, &plan);
        entry.pairing = Some(ModelPairing {
            state: if plan.status == ModelStatus::Ready {
                ModelPairingState::Resolved
            } else {
                ModelPairingState::Unresolved
            },
            checked_projector_index_revision: 0,
            compatible_vision_projector_types: plan.compatible_vision_projector_types.clone(),
            reason: match plan.status {
                ModelStatus::Ready => None,
                ModelStatus::NeedsProjector => Some(ModelPairingReason::NoMatch),
                ModelStatus::Broken => Some(ModelPairingReason::MissingMetadata),
            },
            updated_at_unix_ms: now_unix_ms(),
        });
        self.registry.insert_model(entry)?;
        self.registry.save()?;
        Ok(ResolvedSource { entry_id })
    }
}

fn cached_local_record_matches(
    record: &AssetRecord,
    kind: Option<ModelAssetKind>,
    source_bytes: u64,
    source_path: &Path,
    source_modified_unix_ms: Option<u64>,
) -> bool {
    if kind.is_some_and(|expected| record.kind != expected) || record.bytes != source_bytes {
        return false;
    }

    let AssetSource::Local {
        path: record_source_path,
        modified_unix_ms: record_modified_unix_ms,
    } = &record.source
    else {
        return false;
    };

    same_path(record_source_path, source_path)
        && match (*record_modified_unix_ms, source_modified_unix_ms) {
            (Some(record), Some(source)) => record == source,
            _ => true,
        }
}
