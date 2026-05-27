//! Unit tests for the parent module.

use super::super::*;

#[test]
fn hex_lower_encodes_lowercase_nibbles() {
    assert_eq!(hex_lower(&[0x00, 0x0f, 0xa5, 0xff]), "000fa5ff");
}

#[test]
fn asset_summary_totals_bytes_and_promotes_remote_source() {
    let summary = asset_summary([(7, false), (5, true)]);

    assert_eq!(summary.bytes, 12);
    assert_eq!(summary.source, ModelSourceKind::Remote);
}

#[test]
fn media_marker_for_modality_defaults_only_for_vision_models() {
    assert_eq!(
        media_marker_for_modality(ModelModality::Vision).as_deref(),
        Some(DEFAULT_MEDIA_MARKER)
    );
    assert_eq!(media_marker_for_modality(ModelModality::Text), None);
}

#[test]
fn classified_asset_defaults_missing_inspection_to_unknown() {
    let asset = classified_asset("asset-id", "asset-name", None);

    assert_eq!(asset.asset_id, "asset-id");
    assert_eq!(asset.name, "asset-name");
    assert_eq!(asset.inspection, AssetInspection::unknown());
}

#[test]
fn sorted_model_asset_ids_includes_projector_and_deduplicates() {
    let model_asset_ids = vec![
        "asset-b".to_string(),
        "asset-a".to_string(),
        "asset-b".to_string(),
    ];
    let projector_asset_id = "asset-projector".to_string();

    assert_eq!(
        sorted_model_asset_ids(&model_asset_ids, Some(&projector_asset_id)),
        vec!["asset-a", "asset-b", "asset-projector"]
    );
}

#[test]
fn missing_model_asset_errors_use_shared_messages() {
    let error = missing_model_asset("asset-a");
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "model references missing asset asset-a")
    );

    let error = model_missing_asset("model-a", "asset-a");
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "model model-a references missing asset asset-a")
    );
}

#[test]
fn load_asset_errors_use_shared_messages() {
    let error = missing_load_asset("asset-a");
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "missing asset asset-a")
    );

    let error = missing_projector_load_asset("asset-a");
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "missing projector asset asset-a")
    );

    let error = model_has_no_assets();
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "model has no assets")
    );
}

#[test]
fn manifest_corruption_errors_use_shared_messages() {
    let error = manifest_version_mismatch("manifest", REGISTRY_MANIFEST_VERSION, 2);
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == format!("expected manifest version {REGISTRY_MANIFEST_VERSION}, got 2"))
    );

    validate_registry_manifest_version("manifest", REGISTRY_MANIFEST_VERSION)
        .expect("current manifest version is accepted");
    let error = validate_registry_manifest_version("manifest", REGISTRY_MANIFEST_VERSION - 1)
        .expect_err("old manifest version is rejected");
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == format!("expected manifest version {REGISTRY_MANIFEST_VERSION}, got {}", REGISTRY_MANIFEST_VERSION - 1))
    );

    let error = manifest_key_mismatch("asset", "asset-key", "asset-id");
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "asset key asset-key does not match record id asset-id")
    );

    let error = asset_refcount_mismatch("asset-a", 2, 1);
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "asset asset-a refcount mismatch: stored 2, expected 1")
    );

    let error = empty_asset_id();
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "asset id must not be empty")
    );

    let error = invalid_asset_field("asset-a", "storagePath must not be empty");
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message == "asset asset-a storagePath must not be empty")
    );
}

#[test]
fn asset_refcount_helpers_update_and_reject_bounds() {
    let mut ref_count = 1;
    increment_asset_refcount(&mut ref_count, "asset-a").expect("increment succeeds");
    assert_eq!(ref_count, 2);

    decrement_asset_refcount(&mut ref_count, "asset-a").expect("decrement succeeds");
    assert_eq!(ref_count, 1);

    let mut saturated = u32::MAX;
    let overflow =
        increment_asset_refcount(&mut saturated, "asset-a").expect_err("overflow is rejected");
    assert_eq!(saturated, u32::MAX);
    assert!(
        matches!(overflow, ModelError::StorageCorrupt(message) if message.contains("refcount overflow"))
    );

    let mut empty = 0;
    let underflow =
        decrement_asset_refcount(&mut empty, "asset-a").expect_err("underflow is rejected");
    assert_eq!(empty, 0);
    assert!(
        matches!(underflow, ModelError::StorageCorrupt(message) if message.contains("already zero"))
    );
}

#[test]
fn bump_projector_index_revision_increments_and_rejects_overflow() {
    let mut revision = 41;
    bump_projector_index_revision(&mut revision).expect("revision increments");
    assert_eq!(revision, 42);

    let mut saturated = u64::MAX;
    let error = bump_projector_index_revision(&mut saturated).expect_err("overflow is rejected");
    assert_eq!(saturated, u64::MAX);
    assert!(
        matches!(error, ModelError::StorageCorrupt(message) if message.contains("revision overflow"))
    );
}
