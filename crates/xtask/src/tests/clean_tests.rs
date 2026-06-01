//! Unit tests for the parent module.

use std::path::PathBuf;

use super::first_child_under;

#[test]
fn finds_top_level_child_for_nested_path() {
    let root = PathBuf::from("workspace").join(".build").join("cargo");
    let path = root.join("debug").join("deps").join("xtask.exe");

    assert_eq!(first_child_under(&root, &path), Some(root.join("debug")));
}

#[test]
fn finds_direct_child_path() {
    let root = PathBuf::from("workspace").join(".build").join("cargo");
    let path = root.join("xtask.exe");

    assert_eq!(
        first_child_under(&root, &path),
        Some(root.join("xtask.exe"))
    );
}

#[test]
fn ignores_path_outside_root() {
    let root = PathBuf::from("workspace").join(".build").join("cargo");
    let path = PathBuf::from("workspace").join(".build").join("xtask");

    assert_eq!(first_child_under(&root, &path), None);
}
