use std::path::PathBuf;

use crate::lifecycle::storage::StorageBackend;
use crate::lifecycle::util::{
    missing_load_asset, missing_projector_load_asset, model_has_no_assets,
};
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
            .ok_or_else(model_has_no_assets)?;
        let model_path = self.resolve_entry_asset_path(model_asset, missing_load_asset)?;

        let projector_path = entry
            .projector_asset_id
            .as_ref()
            .map(|asset_id| self.resolve_entry_asset_path(asset_id, missing_projector_load_asset))
            .transpose()?;

        Ok(LoadAssetPaths {
            model_path,
            projector_path,
        })
    }

    fn resolve_entry_asset_path(
        &self,
        asset_id: &str,
        missing_asset: fn(&str) -> ModelError,
    ) -> Result<PathBuf, ModelError> {
        let record = self
            .registry
            .manifest
            .assets
            .get(asset_id)
            .ok_or_else(|| missing_asset(asset_id))?;
        self.assets.resolve_asset_path(record)
    }
}

#[cfg(test)]
#[path = "../../tests/lifecycle/service/load_assets_tests.rs"]
mod load_assets_tests;
