//! Tests the `utils` module in `xtask`.
//!
//! Covers fake workspace path construction, platform-specific tags, formatting,
//! and deterministic child directory discovery without querying Playwright or
//! invoking external commands.

use crate::cli::Backend;
use crate::test_support::TempDir;

use super::{read_child_dirs, BuildContext};

#[test]
fn build_context_paths_are_rooted_under_fake_workspace() {
    let temp = TempDir::new("utils-paths");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    assert_eq!(ctx.build_root(), temp.join(".build"));
    assert_eq!(
        ctx.cargo_node_target_dir(&Backend::Vulkan),
        temp.join(".build/cargo/node/vulkan")
    );
    assert_eq!(
        ctx.cargo_cli_target_dir(&Backend::Cuda),
        temp.join(".build/cargo/cli/cuda")
    );
    assert_eq!(
        ctx.cargo_python_target_dir(None),
        temp.join(".build/cargo/python/cpu")
    );
    assert_eq!(
        ctx.cargo_wasm_target_dir(true),
        temp.join(".build/cargo/wasm/pthread")
    );
    assert_eq!(
        ctx.cmake_llama_build_dir(&Backend::Cpu),
        temp.join(".build/cmake/llama/cpu")
    );
    assert_eq!(ctx.node_artifacts_dir(), temp.join(".build/artifacts/node"));
    assert_eq!(
        ctx.python_artifacts_dir(),
        temp.join(".build/artifacts/python")
    );
    assert_eq!(ctx.cli_artifacts_dir(), temp.join(".build/artifacts/cli"));
    assert_eq!(
        ctx.npm_browser_wasm_dir(),
        temp.join(".build/artifacts/npm/cogentlm/dist/wasm")
    );
    assert_eq!(
        ctx.app_artifacts_dir("examples"),
        temp.join(".build/artifacts/apps/examples")
    );
    assert_eq!(ctx.launcher_bin_dir(), temp.join(".build/bin"));
    assert_eq!(ctx.sample_models_dir(), temp.join(".build/models"));
    assert_eq!(ctx.bindings_node_dir(), temp.join("bindings/node"));
    assert_eq!(ctx.bindings_python_dir(), temp.join("bindings/python"));
}

#[test]
fn build_tags_and_command_paths_are_stable() {
    let temp = TempDir::new("utils-tags");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    assert_eq!(BuildContext::backend_build_tag(None), "cpu");
    assert_eq!(
        BuildContext::backend_build_tag(Some(&Backend::Metal)),
        "metal"
    );
    assert_eq!(BuildContext::wasm_build_tag(false), "single");
    assert_eq!(BuildContext::wasm_build_tag(true), "pthread");
    assert_eq!(
        ctx.command_path(&temp.join("with space")),
        format!("\"{}\"", temp.join("with space").display())
    );
    assert!(!ctx.cmake_file_path(&temp.join("a/b")).contains('\\'));
}

#[test]
fn child_dirs_are_sorted_and_missing_roots_are_empty() {
    let temp = TempDir::new("utils-child-dirs");
    let root = temp.create_dir("children");
    temp.create_dir("children/zeta");
    temp.create_dir("children/alpha");
    temp.write("children/file.txt", "not a directory");

    let dirs = read_child_dirs(&root).unwrap();
    assert_eq!(dirs, vec![root.join("alpha"), root.join("zeta")]);
    assert!(read_child_dirs(&temp.join("missing")).unwrap().is_empty());
}
