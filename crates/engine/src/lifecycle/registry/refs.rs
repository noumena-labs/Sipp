use std::collections::BTreeMap;

use crate::collection::sorted_ref_deltas;
use crate::lifecycle::util::{
    asset_refcount_mismatch, bump_projector_index_revision, decrement_asset_refcount,
    increment_asset_refcount, increment_expected_asset_refcount, manifest_key_mismatch,
    missing_model_asset, model_missing_asset, sorted_model_asset_ids,
    validate_registry_manifest_version,
};
use crate::lifecycle::{ModelEntry, ModelError, RegistryManifest};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/registry/refs_tests.rs"]
mod refs_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(super) fn validate_manifest(manifest: &RegistryManifest) -> Result<(), ModelError> {
    validate_registry_manifest_version("manifest", manifest.version)?;
    let mut expected_ref_counts = BTreeMap::<String, u32>::new();
    for (id, asset) in &manifest.assets {
        if id != &asset.id {
            return Err(manifest_key_mismatch("asset", id, &asset.id));
        }
    }
    for (id, model) in &manifest.models {
        if id != &model.id {
            return Err(manifest_key_mismatch("model", id, &model.id));
        }
        for asset_id in referenced_asset_ids(model) {
            if !manifest.assets.contains_key(&asset_id) {
                return Err(model_missing_asset(id, &asset_id));
            }
            increment_expected_asset_refcount(&mut expected_ref_counts, &asset_id)?;
        }
    }
    for (id, asset) in &manifest.assets {
        let expected = expected_ref_counts.get(id).copied().unwrap_or(0);
        if asset.ref_count != expected {
            return Err(asset_refcount_mismatch(id, asset.ref_count, expected));
        }
    }
    Ok(())
}

pub(super) fn referenced_asset_ids(model: &ModelEntry) -> Vec<String> {
    sorted_model_asset_ids(&model.model_asset_ids, model.projector_asset_id.as_ref())
}

pub(super) fn increment_refs(
    manifest: &mut RegistryManifest,
    asset_ids: Vec<String>,
) -> Result<(), ModelError> {
    adjust_refs(manifest, asset_ids, increment_asset_refcount)
}

pub(super) fn rebalance_refs(
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

pub(super) fn decrement_refs(
    manifest: &mut RegistryManifest,
    asset_ids: Vec<String>,
) -> Result<(), ModelError> {
    adjust_refs(manifest, asset_ids, decrement_asset_refcount)
}

fn adjust_refs(
    manifest: &mut RegistryManifest,
    asset_ids: Vec<String>,
    adjust_refcount: fn(&mut u32, &str) -> Result<(), ModelError>,
) -> Result<(), ModelError> {
    for id in asset_ids {
        let asset = manifest
            .assets
            .get_mut(&id)
            .ok_or_else(|| missing_model_asset(&id))?;
        adjust_refcount(&mut asset.ref_count, &id)?;
    }
    bump_manifest_projector_index_revision(manifest)?;
    Ok(())
}

fn bump_manifest_projector_index_revision(
    manifest: &mut RegistryManifest,
) -> Result<(), ModelError> {
    bump_projector_index_revision(&mut manifest.projector_index_revision)
}
