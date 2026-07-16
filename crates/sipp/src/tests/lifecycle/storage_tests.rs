//! Tests the `lifecycle::storage` module in `sipp`.
//!
//! Covers lifecycle registry, storage, browser, service, and pairing behavior with temporary storage and pure fixtures instead of native runtime loading.

use super::*;
use crate::lifecycle::test_support::TempDir;
use std::fs;

fn unsupported_version_gguf() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x4655_4747_u32.to_le_bytes());
    bytes.extend_from_slice(&99_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes
}

#[test]
fn asset_store_hashes_and_dedupes_local_files() {
    let root = TempDir::new("storage", "dedupe");
    let source = root.path.join("source.gguf");
    fs::write(&source, b"not a real gguf, just stable bytes").expect("source");

    let store = AssetStore::local(root.path.join("store"));
    let first = store
        .install_local_path_as(&source, None)
        .expect("first install");
    let second = store
        .install_local_path_as(&source, None)
        .expect("second install");

    assert_eq!(first.record.id, second.record.id);
    assert!(!first.already_present);
    assert!(second.already_present);
    assert_eq!(first.record.bytes, 34);
    assert!(matches!(
        second.record.source,
        AssetSource::Local {
            path: _,
            modified_unix_ms: Some(_)
        }
    ));
    assert!(store
        .resolve_asset_path(&first.record)
        .expect("asset")
        .exists());
}

#[test]
fn existing_asset_path_must_match_source_hash() {
    let root = TempDir::new("storage", "corrupt-existing");
    let source = root.path.join("source.gguf");
    fs::write(&source, b"stable source bytes").expect("source");

    let store = AssetStore::local(root.path.join("store"));
    let installed = store.install_local_path_as(&source, None).expect("install");
    let asset_path = store.resolve_asset_path(&installed.record).expect("asset");
    fs::remove_file(&asset_path).expect("remove linked asset");
    fs::write(asset_path, b"different bytes now").expect("corrupt same len");

    let error = store
        .install_local_path_as(&source, None)
        .expect_err("corrupt existing asset");

    assert!(matches!(error, ModelError::StorageCorrupt(_)));
}

#[test]
fn missing_asset_is_typed_error() {
    let root = TempDir::new("storage", "missing");
    let source = root.path.join("source.gguf");
    fs::write(&source, b"bytes").expect("source");

    let store = AssetStore::local(root.path.join("store"));
    let installed = store.install_local_path_as(&source, None).expect("install");
    store.delete_asset(&installed.record).expect("delete");

    let error = store
        .resolve_asset_path(&installed.record)
        .expect_err("missing asset");
    assert!(matches!(error, ModelError::AssetMissing(_)));
}

#[test]
fn remote_staged_inspection_failure_keeps_staged_file_unpublished() {
    let root = TempDir::new("storage", "remote-invalid-gguf");
    let staged_path = root.path.join("download.gguf");
    let staged_bytes = unsupported_version_gguf();
    fs::write(&staged_path, &staged_bytes).expect("staged download");

    let store = AssetStore::local(root.path.join("store"));
    let metadata = crate::lifecycle::acquisition::RemoteMetadata {
        url: "https://example.test/model.gguf".to_string(),
        name: "model.gguf".to_string(),
        bytes: u64::try_from(staged_bytes.len()).expect("staged byte length"),
        etag: None,
        last_modified: None,
    };

    let error = store
        .install_remote_staged(&staged_path, &metadata, ModelAssetKind::Model)
        .expect_err("unsupported GGUF version");

    assert!(matches!(error, ModelError::UnsupportedGgufVersion(99)));
    assert!(staged_path.exists());
    assert_eq!(
        fs::read_dir(root.path.join("store").join("assets"))
            .expect("assets directory")
            .count(),
        0
    );
}
