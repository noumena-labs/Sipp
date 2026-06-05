//! Tests the `clean` module in `xtask`.
//!
//! Covers cleanup target selection, generated directory expansion, path safety,
//! and purge/toolchain edge cases with fake workspaces instead of deleting real
//! developer artifacts.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use crate::cli::CleanArgs;
use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{add_generated_dirs, checked_delete_target, clean_targets, first_child_under};

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

#[test]
fn generated_dirs_include_every_workspace_output_name() {
    let mut targets = BTreeSet::new();
    let root = PathBuf::from("app");

    add_generated_dirs(&mut targets, &root);

    for name in [
        "dist", ".vite", ".turbo", "coverage", "build", "out", ".next",
    ] {
        assert!(targets.contains(&root.join(name)));
    }
}

#[test]
fn clean_targets_include_generated_dirs_and_optional_purge_roots() {
    let temp = TempDir::new("clean-targets");
    temp.create_dir("demos/chat");
    temp.create_dir("benchmarks/browser");
    temp.create_dir("lib/web");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    let base = clean_targets(
        &ctx,
        &CleanArgs {
            purge: false,
            toolchains: false,
            dry_run: true,
        },
    )
    .unwrap();
    assert!(base.contains(&ctx.cargo_build_root()));
    assert!(base.contains(&ctx.demo_dir("chat").join("dist")));
    assert!(base.contains(&ctx.benchmark_browser_dir().join(".vite")));
    assert!(base.contains(&ctx.browser_package_dir().join(".turbo")));
    assert!(!base.contains(&ctx.workspace_root().join("node_modules")));
    assert!(!base.contains(&ctx.toolchain_dir()));

    let expanded = clean_targets(
        &ctx,
        &CleanArgs {
            purge: true,
            toolchains: true,
            dry_run: true,
        },
    )
    .unwrap();
    assert!(expanded.contains(&ctx.workspace_root().join("node_modules")));
    assert!(expanded.contains(&ctx.bindings_node_dir().join("node_modules")));
    assert!(expanded.contains(&ctx.demo_dir("chat").join("node_modules")));
    assert!(expanded.contains(&ctx.benchmark_browser_dir().join("node_modules")));
    assert!(expanded.contains(&ctx.toolchain_dir()));
}

#[test]
fn checked_delete_target_accepts_children_and_rejects_workspace_or_outside_paths() {
    let temp = TempDir::new("clean-delete");
    let workspace = fs::canonicalize(temp.path()).unwrap();
    let child = temp.create_dir("child");
    let outside = TempDir::new("clean-delete-outside");

    assert_eq!(
        checked_delete_target(&workspace, &child).unwrap(),
        fs::canonicalize(&child).unwrap()
    );
    assert!(checked_delete_target(&workspace, temp.path()).is_err());
    assert!(checked_delete_target(&workspace, outside.path()).is_err());
}
