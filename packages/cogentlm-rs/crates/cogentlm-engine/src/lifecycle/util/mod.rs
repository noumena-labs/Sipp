//! Small filesystem and hashing helpers shared across the lifecycle module.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::collection::sorted_unique_strings_with_optional;

use super::{
    AssetInspection, ClassifiedAsset, ModelError, ModelModality, ModelSourceKind,
    DEFAULT_MEDIA_MARKER, REGISTRY_MANIFEST_VERSION,
};

const MODEL_ID_PREFIX: &str = "model-";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct AssetSummary {
    pub(super) source: ModelSourceKind,
    pub(super) bytes: u64,
}

pub(super) fn asset_summary(assets: impl IntoIterator<Item = (u64, bool)>) -> AssetSummary {
    let mut summary = AssetSummary {
        source: ModelSourceKind::Local,
        bytes: 0,
    };

    for (bytes, remote) in assets {
        debug_assert!(summary.bytes.checked_add(bytes).is_some());
        summary.bytes = summary.bytes.saturating_add(bytes);
        if remote {
            summary.source = ModelSourceKind::Remote;
        }
    }

    summary
}

pub(super) fn media_marker_for_modality(modality: ModelModality) -> Option<String> {
    (modality == ModelModality::Vision).then(|| DEFAULT_MEDIA_MARKER.to_string())
}

pub(super) fn classified_asset(
    asset_id: impl Into<String>,
    name: impl Into<String>,
    inspection: Option<AssetInspection>,
) -> ClassifiedAsset {
    ClassifiedAsset {
        asset_id: asset_id.into(),
        name: name.into(),
        inspection: inspection.unwrap_or_else(AssetInspection::unknown),
    }
}

pub(super) fn sorted_model_asset_ids(
    model_asset_ids: &[String],
    projector_asset_id: Option<&String>,
) -> Vec<String> {
    sorted_unique_strings_with_optional(model_asset_ids.to_vec(), projector_asset_id)
}

pub(super) fn model_id_from_fingerprint(fingerprint: &str) -> String {
    format!("{MODEL_ID_PREFIX}{fingerprint}")
}

pub(super) fn model_id_from_fingerprint_prefix(fingerprint: &str, prefix_chars: usize) -> String {
    format!("{MODEL_ID_PREFIX}{}", &fingerprint[..prefix_chars])
}

pub(super) fn storage_corrupt(message: impl Into<String>) -> ModelError {
    ModelError::StorageCorrupt(message.into())
}

pub(super) fn missing_model_asset(asset_id: &str) -> ModelError {
    storage_corrupt(format!("model references missing asset {asset_id}"))
}

pub(super) fn model_missing_asset(model_id: &str, asset_id: &str) -> ModelError {
    storage_corrupt(format!(
        "model {model_id} references missing asset {asset_id}"
    ))
}

pub(super) fn missing_load_asset(asset_id: &str) -> ModelError {
    storage_corrupt(format!("missing asset {asset_id}"))
}

pub(super) fn missing_projector_load_asset(asset_id: &str) -> ModelError {
    storage_corrupt(format!("missing projector asset {asset_id}"))
}

pub(super) fn model_has_no_assets() -> ModelError {
    storage_corrupt("model has no assets")
}

pub(super) fn manifest_version_mismatch(label: &str, expected: u32, actual: u32) -> ModelError {
    storage_corrupt(format!("expected {label} version {expected}, got {actual}"))
}

pub(super) fn validate_registry_manifest_version(
    label: &str,
    actual: u32,
) -> Result<(), ModelError> {
    if actual == REGISTRY_MANIFEST_VERSION {
        return Ok(());
    }

    Err(manifest_version_mismatch(
        label,
        REGISTRY_MANIFEST_VERSION,
        actual,
    ))
}

pub(super) fn manifest_key_mismatch(kind: &str, key: &str, record_id: &str) -> ModelError {
    storage_corrupt(format!(
        "{kind} key {key} does not match record id {record_id}"
    ))
}

pub(super) fn asset_refcount_mismatch(asset_id: &str, stored: u32, expected: u32) -> ModelError {
    storage_corrupt(format!(
        "asset {asset_id} refcount mismatch: stored {stored}, expected {expected}"
    ))
}

pub(super) fn empty_asset_id() -> ModelError {
    storage_corrupt("asset id must not be empty")
}

pub(super) fn invalid_asset_field(asset_id: &str, message: &str) -> ModelError {
    storage_corrupt(format!("asset {asset_id} {message}"))
}

pub(super) fn model_not_found(model_id: &str) -> ModelError {
    ModelError::ModelNotFound(model_id.to_string())
}

pub(super) fn invalid_pairing(message: impl Into<String>) -> ModelError {
    ModelError::InvalidModelPairing(message.into())
}

pub(super) fn invalid_source(message: impl Into<String>) -> ModelError {
    ModelError::InvalidModelSource(message.into())
}

pub(super) fn asset_refcount_overflow(asset_id: &str) -> ModelError {
    storage_corrupt(format!("asset {asset_id} refcount overflow"))
}

pub(super) fn increment_asset_refcount(
    ref_count: &mut u32,
    asset_id: &str,
) -> Result<(), ModelError> {
    *ref_count = ref_count
        .checked_add(1)
        .ok_or_else(|| asset_refcount_overflow(asset_id))?;
    Ok(())
}

pub(super) fn increment_expected_asset_refcount(
    expected_ref_counts: &mut BTreeMap<String, u32>,
    asset_id: &str,
) -> Result<(), ModelError> {
    increment_asset_refcount(
        expected_ref_counts.entry(asset_id.to_string()).or_default(),
        asset_id,
    )
}

pub(super) fn decrement_asset_refcount(
    ref_count: &mut u32,
    asset_id: &str,
) -> Result<(), ModelError> {
    if *ref_count == 0 {
        return Err(storage_corrupt(format!(
            "asset {asset_id} refcount is already zero"
        )));
    }
    *ref_count -= 1;
    Ok(())
}

pub(super) fn bump_projector_index_revision(revision: &mut u64) -> Result<(), ModelError> {
    *revision = revision
        .checked_add(1)
        .ok_or_else(|| storage_corrupt("projector index revision overflow"))?;
    Ok(())
}

pub(crate) fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let capacity = bytes.len().checked_mul(2).unwrap_or(bytes.len());
    let mut output = String::with_capacity(capacity);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex_lower(&Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    mod util_tests;
}
