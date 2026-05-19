//! Unit tests for the parent module.

use super::*;
use std::fs;

struct TempDir {
    path: PathBuf,
}

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "cogentlm-core-storage-{}-{}",
            name,
            now_unix_ms()
        ));
        fs::create_dir_all(&path).expect("temp dir");
        Self { path }
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

#[test]
fn asset_store_hashes_and_dedupes_local_files() {
    let root = TempDir::new("dedupe");
    let source = root.path.join("source.gguf");
    fs::write(&source, b"not a real gguf, just stable bytes").expect("source");

    let store = AssetStore::local(root.path.join("store"));
    let first = store.install_local_path(&source).expect("first install");
    let second = store.install_local_path(&source).expect("second install");

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
    let root = TempDir::new("corrupt-existing");
    let source = root.path.join("source.gguf");
    fs::write(&source, b"stable source bytes").expect("source");

    let store = AssetStore::local(root.path.join("store"));
    let installed = store.install_local_path(&source).expect("install");
    let asset_path = store.resolve_asset_path(&installed.record).expect("asset");
    fs::remove_file(&asset_path).expect("remove linked asset");
    fs::write(asset_path, b"different bytes now").expect("corrupt same len");

    let error = store
        .install_local_path(&source)
        .expect_err("corrupt existing asset");

    assert!(matches!(error, ModelError::StorageCorrupt(_)));
}

#[test]
fn missing_asset_is_typed_error() {
    let root = TempDir::new("missing");
    let source = root.path.join("source.gguf");
    fs::write(&source, b"bytes").expect("source");

    let store = AssetStore::local(root.path.join("store"));
    let installed = store.install_local_path(&source).expect("install");
    store.delete_asset(&installed.record).expect("delete");

    let error = store
        .resolve_asset_path(&installed.record)
        .expect_err("missing asset");
    assert!(matches!(error, ModelError::AssetMissing(_)));
}

#[test]
fn inspection_prefix_capacity_clamps_to_prefix_limit() {
    assert_eq!(inspection_prefix_capacity(0), 0);
    assert_eq!(inspection_prefix_capacity(1024), 1024);
    assert_eq!(
        inspection_prefix_capacity((INSPECTION_PREFIX_BYTES as u64) + 1),
        INSPECTION_PREFIX_BYTES
    );
    assert_eq!(
        inspection_prefix_capacity(u64::MAX),
        INSPECTION_PREFIX_BYTES
    );
}
