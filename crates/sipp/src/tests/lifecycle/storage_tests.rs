//! Tests the `lifecycle::storage` module in `sipp`.
//!
//! Covers lifecycle registry, storage, browser, service, and pairing behavior with temporary storage and pure fixtures instead of native runtime loading.

use super::*;
use crate::lifecycle::test_support::TempDir;
use crate::lifecycle::RegistryManifest;
use std::fs;
use std::path::PathBuf;

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
        .install_remote_staged(&staged_path, &metadata, ModelAssetKind::Model, None)
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

#[test]
fn acquisition_journal_recovery_removes_only_unregistered_paths() {
    let root = TempDir::new("storage", "journal-recovery");
    let store_root = root.path.join("store");
    let store = AssetStore::local(&store_root);
    let orphan_storage_path = PathBuf::from("assets").join("asset-orphan");
    let keep_storage_path = PathBuf::from("assets").join("asset-keep");
    let orphan_path = store_root.join(&orphan_storage_path);
    let keep_path = store_root.join(&keep_storage_path);
    fs::create_dir_all(store_root.join("assets")).expect("assets directory");
    fs::write(&orphan_path, b"orphan").expect("orphan asset");
    fs::write(&keep_path, b"keep").expect("registered asset");

    let journal = store.acquisition_journal("lease-1");
    journal
        .record_path(&orphan_storage_path)
        .expect("record orphan");
    journal
        .record_path(&keep_storage_path)
        .expect("record keep");

    let mut manifest = RegistryManifest::default();
    manifest.assets.insert(
        "asset-keep".to_string(),
        AssetRecord {
            id: "asset-keep".to_string(),
            kind: ModelAssetKind::Model,
            name: "model.gguf".to_string(),
            hash: "hash".to_string(),
            bytes: 4,
            storage_path: keep_storage_path,
            source: AssetSource::Remote {
                url: "https://example.test/model.gguf".to_string(),
                etag: None,
                last_modified: None,
            },
            ref_count: 1,
            created_at_unix_ms: 0,
            inspection: None,
        },
    );

    store
        .recover_acquisition_journals(&manifest)
        .expect("recover journal");

    assert!(!orphan_path.exists());
    assert!(keep_path.exists());
    assert_eq!(
        fs::read_dir(store_root.join(".incoming").join("journals"))
            .expect("journal directory")
            .count(),
        0
    );
}
