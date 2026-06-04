//! Tests the `build_support::util` module in `cogentlm-sys`.
//!
//! Covers deterministic path sanitization and path-component normalization
//! without invoking native build tools.

use std::path::PathBuf;

use super::*;

#[test]
fn sanitize_path_strips_windows_verbatim_prefix() {
    assert_eq!(
        sanitize_path(r"\\?\workspace\crates\sys"),
        PathBuf::from(r"workspace\crates\sys")
    );
}

#[test]
fn sanitize_path_preserves_normal_paths() {
    assert_eq!(
        sanitize_path("workspace/crates/sys"),
        PathBuf::from("workspace/crates/sys")
    );
}

#[test]
fn path_component_uses_fallback_for_empty_values() {
    assert_eq!(path_component("", "host"), "host");
}

#[test]
fn path_component_replaces_filesystem_separators_and_colons() {
    assert_eq!(
        path_component("x86_64/pc\\windows:msvc", "host"),
        "x86_64_pc_windows_msvc"
    );
}
