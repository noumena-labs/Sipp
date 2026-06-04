//! Tests the `lifecycle::storage::metadata` module in `cogentlm-engine`.
//!
//! Covers asset-name normalization and temporary suffix generation using
//! deterministic path/string assertions and local metadata only.

use std::fs;
use std::path::Path;

use crate::lifecycle::test_support::TempDir;

use super::*;

#[test]
fn normalize_asset_name_sanitizes_filesystem_reserved_characters() {
    let name = normalize_asset_name(Path::new(r#"bad:name*with?chars".gguf"#));

    assert_eq!(name, "bad-name-with-chars-.gguf");
}

#[test]
fn normalize_asset_name_uses_default_for_missing_or_blank_names() {
    assert_eq!(normalize_asset_name(Path::new("")), DEFAULT_MODEL_FILE_NAME);
    assert_eq!(
        normalize_asset_name(Path::new("   ")),
        DEFAULT_MODEL_FILE_NAME
    );
}

#[test]
fn unique_temp_suffix_is_monotonic_enough_for_same_process_names() {
    let first = unique_temp_suffix();
    let second = unique_temp_suffix();

    assert_ne!(first, second);
    assert!(first.contains('-'));
    assert!(second.contains('-'));
}

#[test]
fn modified_unix_ms_returns_file_timestamp_when_available() {
    let root = TempDir::new("metadata", "modified");
    let path = root.path.join("model.gguf");
    fs::write(&path, b"model").expect("file");
    let metadata = fs::metadata(path).expect("metadata");

    assert!(modified_unix_ms(&metadata).is_some());
}
