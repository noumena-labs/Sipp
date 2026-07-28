//! Tests the `lifecycle::storage` module in `sipp`.
//!
//! Covers local references, managed remote assets, and acquisition recovery.

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
fn asset_store_references_local_files() {
    let root = TempDir::new("storage", "local-reference");
    let source = root.path.join("source.gguf");
    fs::write(&source, b"not a real gguf, just stable bytes").expect("source");

    let store_root = root.path.join("store");
    let store = AssetStore::local(&store_root);
    let first = store.register_local_path(&source).expect("first add");
    let second = store.register_local_path(&source).expect("second add");
    let canonical_source = fs::canonicalize(&source).expect("canonical source");

    assert_eq!(first.id, second.id);
    assert_eq!(first.storage_path, canonical_source);
    assert!(matches!(
        &second.source,
        AssetSource::Local { path, .. } if path == &canonical_source
    ));
    assert_eq!(
        store.resolve_asset_path(&first).expect("asset"),
        canonical_source
    );
    assert!(!store_root.join("assets").exists());
}

#[test]
fn local_asset_identity_depends_only_on_content() {
    let root = TempDir::new("storage", "content-identity");
    let first_path = root.path.join("first.gguf");
    let second_path = root.path.join("second.gguf");
    fs::write(&first_path, b"identical model bytes").expect("first source");
    fs::write(&second_path, b"identical model bytes").expect("second source");

    let store = AssetStore::local(root.path.join("store"));
    let first = store.register_local_path(first_path).expect("first add");
    let second = store.register_local_path(second_path).expect("second add");

    assert_eq!(first.id, second.id);
    assert_eq!(first.hash, second.hash);
}

#[test]
fn source_root_relative_asset_survives_container_relocation() {
    let root = TempDir::new("storage", "source-root-relocation");
    let first_container = root.path.join("container-a");
    let first_source_root = first_container.join("home");
    let first_store_root = first_source_root.join("Library").join("Sipp");
    let first_source = first_source_root.join("Documents").join("model.gguf");
    fs::create_dir_all(first_source.parent().expect("source parent")).expect("source directory");
    fs::write(&first_source, b"relocatable model bytes").expect("source");

    let first_store = AssetStore::new(LocalStorageBackend::with_local_source_root(
        &first_store_root,
        &first_source_root,
    ));
    let record = first_store
        .register_local_path(&first_source)
        .expect("register source-root asset");

    assert_eq!(
        record.storage_path,
        PathBuf::from("Documents").join("model.gguf")
    );
    assert!(matches!(
        &record.source,
        AssetSource::Local {
            path,
            anchor: LocalPathAnchor::SourceRoot,
            ..
        } if path == &PathBuf::from("Documents").join("model.gguf")
    ));

    let second_container = root.path.join("container-b");
    fs::rename(&first_container, &second_container).expect("relocate container");
    let second_source_root = second_container.join("home");
    let second_store_root = second_source_root.join("Library").join("Sipp");
    let second_store = AssetStore::new(LocalStorageBackend::with_local_source_root(
        second_store_root,
        &second_source_root,
    ));

    assert_eq!(
        second_store
            .resolve_asset_path(&record)
            .expect("relocated asset"),
        second_source_root.join("Documents").join("model.gguf")
    );
}

#[test]
fn deleting_a_local_registration_does_not_delete_its_source() {
    let root = TempDir::new("storage", "delete-local-registration");
    let source = root.path.join("source.gguf");
    fs::write(&source, b"stable source bytes").expect("source");

    let store = AssetStore::local(root.path.join("store"));
    let record = store.register_local_path(&source).expect("add");

    store
        .delete_managed_asset(&record)
        .expect("remove registration");

    assert!(source.exists());
}

#[test]
fn missing_local_source_is_typed_error() {
    let root = TempDir::new("storage", "missing");
    let source = root.path.join("source.gguf");
    fs::write(&source, b"bytes").expect("source");

    let store = AssetStore::local(root.path.join("store"));
    let record = store.register_local_path(&source).expect("add");
    fs::remove_file(&source).expect("remove source");

    let error = store
        .resolve_asset_path(&record)
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
        .install_remote_staged(&staged_path, &metadata, None)
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

#[test]
fn acquisition_recovery_does_not_treat_local_locators_as_managed_assets() {
    let root = TempDir::new("storage", "journal-local-locator");
    let store_root = root.path.join("store");
    let store = AssetStore::local(&store_root);
    let staged_storage_path = PathBuf::from("Documents").join("model.gguf");
    let staged_path = store_root.join(&staged_storage_path);
    fs::create_dir_all(staged_path.parent().expect("staged parent")).expect("staged directory");
    fs::write(&staged_path, b"uncommitted").expect("staged asset");

    let journal = store.acquisition_journal("lease-local-locator");
    journal
        .record_path(&staged_storage_path)
        .expect("record staged path");

    let mut manifest = RegistryManifest::default();
    manifest.assets.insert(
        "asset-local".to_string(),
        AssetRecord {
            id: "asset-local".to_string(),
            kind: ModelAssetKind::Model,
            name: "model.gguf".to_string(),
            hash: "hash".to_string(),
            bytes: 1,
            storage_path: staged_storage_path.clone(),
            source: AssetSource::Local {
                path: staged_storage_path,
                anchor: LocalPathAnchor::SourceRoot,
                modified_unix_ms: None,
            },
            ref_count: 0,
            created_at_unix_ms: 0,
            inspection: None,
        },
    );

    store
        .recover_acquisition_journals(&manifest)
        .expect("recover journal");

    assert!(!staged_path.exists());
}
