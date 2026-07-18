//! Tests the `lifecycle::service::source_resolution` module in
//! `sipp`.
//!
//! Covers deterministic source-resolution error paths, cached-local matching,
//! and remote-unavailable branches with temporary local fixtures only.

use std::path::{Path, PathBuf};

use crate::lifecycle::test_support::{gguf_name, TempDir};
use crate::lifecycle::{AssetRecord, AssetSource, ModelStore};
use futures::executor::block_on;

use super::*;

fn local_record(id: &str, path: impl Into<PathBuf>, bytes: u64) -> AssetRecord {
    AssetRecord {
        id: id.to_string(),
        kind: ModelAssetKind::Model,
        name: gguf_name(id),
        hash: id.to_string(),
        bytes,
        storage_path: PathBuf::from("assets").join(id),
        source: AssetSource::Local {
            path: path.into(),
            modified_unix_ms: Some(5),
        },
        ref_count: 0,
        created_at_unix_ms: 0,
        inspection: None,
    }
}

#[test]
fn remove_rejects_missing_model_id() {
    let root = TempDir::new("source-resolution", "missing-installed");
    let store = ModelStore::local(root.path.join("store")).expect("store");

    let error = block_on(store.remove("missing")).expect_err("missing installed model");

    assert!(matches!(error, ModelError::ModelNotFound(id) if id == "missing"));
}

#[test]
fn empty_model_paths_are_invalid_before_storage_access() {
    let root = TempDir::new("source-resolution", "empty-paths");
    let store = ModelStore::local(root.path.join("store")).expect("store");

    let error = block_on(store.install_files(Vec::<PathBuf>::new())).expect_err("empty paths");

    assert!(
        matches!(error, ModelError::InvalidModelSource(message) if message == MODEL_PATHS_REQUIRED)
    );
}

#[test]
fn invalid_remote_url_is_rejected_before_http_access() {
    let root = TempDir::new("source-resolution", "remote");
    let store = ModelStore::local(root.path.join("store")).expect("store");

    let error =
        block_on(store.install_urls(["file:///model.gguf"])).expect_err("invalid remote URL");

    assert!(matches!(error, ModelError::InvalidModelSource(message) if message.contains("http")));
}

#[test]
fn cached_local_record_matching_checks_kind_size_source_and_timestamp() {
    let record = local_record("asset-a", PathBuf::from("model.gguf"), 10);
    let source_path = Path::new("model.gguf");

    assert!(cached_local_record_matches(
        &record,
        Some(ModelAssetKind::Model),
        10,
        source_path,
        Some(5)
    ));
    assert!(!cached_local_record_matches(
        &record,
        Some(ModelAssetKind::Projector),
        10,
        source_path,
        Some(5)
    ));
    assert!(!cached_local_record_matches(
        &record,
        Some(ModelAssetKind::Model),
        11,
        source_path,
        Some(5)
    ));
    assert!(!cached_local_record_matches(
        &record,
        Some(ModelAssetKind::Model),
        10,
        source_path,
        Some(6)
    ));
}

#[test]
fn cached_local_record_matching_rejects_remote_sources() {
    let mut record = local_record("asset-a", PathBuf::from("model.gguf"), 10);
    record.source = AssetSource::Remote {
        url: "https://example.test/model.gguf".to_string(),
        etag: None,
        last_modified: None,
    };

    assert!(!cached_local_record_matches(
        &record,
        Some(ModelAssetKind::Model),
        10,
        Path::new("model.gguf"),
        None
    ));
}

#[test]
fn directory_asset_path_is_rejected_as_invalid_source() {
    let root = TempDir::new("source-resolution", "directory");
    let store = ModelStore::local(root.path.join("store")).expect("store");

    let error = block_on(store.install_files([root.path.clone()])).expect_err("directory source");

    assert!(
        matches!(error, ModelError::InvalidModelSource(message) if message.contains("not a file"))
    );
}
