use std::collections::BTreeMap;

use crate::lifecycle::{ModelEntry, ModelError, RegistryManifest};

pub(super) fn validate_manifest(manifest: &RegistryManifest) -> Result<(), ModelError> {
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

pub(super) fn referenced_asset_ids(model: &ModelEntry) -> Vec<String> {
    let mut ids = model.model_asset_ids.clone();
    if let Some(projector_id) = &model.projector_asset_id {
        ids.push(projector_id.clone());
    }
    ids.sort();
    ids.dedup();
    ids
}

pub(super) fn increment_refs(
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

pub(super) fn decrement_refs(
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
mod tests {
    use super::*;

    #[test]
    fn sorted_ref_deltas_reports_linear_adds_and_removals() {
        let previous = vec![
            "asset-a".to_string(),
            "asset-c".to_string(),
            "asset-e".to_string(),
        ];
        let updated = vec![
            "asset-b".to_string(),
            "asset-c".to_string(),
            "asset-f".to_string(),
        ];

        let (removed, added) = sorted_ref_deltas(&previous, &updated);

        assert_eq!(removed, vec!["asset-a", "asset-e"]);
        assert_eq!(added, vec!["asset-b", "asset-f"]);
    }

    #[test]
    fn sorted_ref_deltas_reports_no_changes_for_equal_refs() {
        let refs = vec!["asset-a".to_string(), "asset-b".to_string()];

        let (removed, added) = sorted_ref_deltas(&refs, &refs);

        assert!(removed.is_empty());
        assert!(added.is_empty());
    }
}
