//! Tests the `lifecycle::service::source_resolution` module in
//! `sipp`.
//!
//! Covers deterministic source classification before acquisition starts.

use std::path::PathBuf;

use crate::lifecycle::test_support::TempDir;
use crate::lifecycle::ModelStore;
use futures::executor::block_on;

use super::*;

#[test]
fn remove_rejects_missing_model_id() {
    let root = TempDir::new("source-resolution", "missing-installed");
    let store = ModelStore::local(root.path.join("store")).expect("store");

    let error = block_on(store.remove("missing")).expect_err("missing installed model");

    assert!(matches!(error, ModelError::ModelNotFound(id) if id == "missing"));
}

#[test]
fn empty_sources_are_invalid_before_storage_access() {
    let root = TempDir::new("source-resolution", "empty-paths");
    let store = ModelStore::local(root.path.join("store")).expect("store");

    let error = block_on(store.add(Vec::<PathBuf>::new())).expect_err("empty sources");

    assert!(
        matches!(error, ModelError::InvalidModelSource(message) if message == MODEL_SOURCES_REQUIRED)
    );
}

#[test]
fn invalid_remote_url_is_rejected_before_http_access() {
    let root = TempDir::new("source-resolution", "remote");
    let store = ModelStore::local(root.path.join("store")).expect("store");

    let error = block_on(store.add(["file:///model.gguf"])).expect_err("invalid remote URL");

    assert!(matches!(error, ModelError::InvalidModelSource(message) if message.contains("http")));
}

#[test]
fn mixed_sources_are_rejected_before_file_or_network_access() {
    let root = TempDir::new("source-resolution", "mixed");
    let store = ModelStore::local(root.path.join("store")).expect("store");

    let error = block_on(store.add(["missing.gguf", "https://example.test/model.gguf"]))
        .expect_err("mixed sources");

    assert!(matches!(
        error,
        ModelError::InvalidModelSource(message) if message.contains("cannot be added together")
    ));
}

#[test]
fn directory_asset_path_is_rejected_as_invalid_source() {
    let root = TempDir::new("source-resolution", "directory");
    let store = ModelStore::local(root.path.join("store")).expect("store");

    let error = block_on(store.add([root.path.clone()])).expect_err("directory source");

    assert!(
        matches!(error, ModelError::InvalidModelSource(message) if message.contains("not a file"))
    );
}
