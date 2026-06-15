//! Tests the `targets::node` module in `xtask`.
//!
//! Covers backend expansion, artifact discovery, and stale artifact cleanup with
//! fixture directories instead of running Bun or napi-rs.

use crate::cli::Backend;
use crate::test_support::{EnvGuard, TempDir};
use crate::utils::BuildContext;

use super::{
    backend_label, backends_to_build, find_artifact, prepare_dist_dir, require_all_backends,
};

#[test]
fn backend_expansion_uses_default_specific_and_host_all_modes() {
    assert_eq!(backends_to_build(None), vec![Backend::Cpu]);
    assert_eq!(
        backends_to_build(Some(&Backend::Metal)),
        vec![Backend::Metal]
    );
    if cfg!(target_os = "macos") {
        assert_eq!(
            backends_to_build(Some(&Backend::All)),
            vec![Backend::Cpu, Backend::Metal]
        );
    } else {
        assert_eq!(
            backends_to_build(Some(&Backend::All)),
            vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda]
        );
    }
    assert_eq!(backend_label(None), "cpu (default)");
    assert_eq!(backend_label(Some(&Backend::Cuda)), "cuda");
}

#[test]
fn artifact_discovery_returns_first_node_file_only() {
    let temp = TempDir::new("target-node-artifact");
    temp.write("staging/readme.txt", "");
    let artifact = temp.write("staging/sipp_node.win32-x64-msvc.node", "");

    assert_eq!(
        find_artifact(&temp.join("staging")).unwrap(),
        Some(artifact)
    );
}

#[test]
fn strict_backend_policy_requires_exact_env_value() {
    let _env = EnvGuard::new(&[("SIPP_REQUIRE_ALL_BACKENDS", None)]);
    assert!(!require_all_backends());
    drop(_env);

    let _env = EnvGuard::new(&[("SIPP_REQUIRE_ALL_BACKENDS", Some("true"))]);
    assert!(!require_all_backends());
    drop(_env);

    let _env = EnvGuard::new(&[("SIPP_REQUIRE_ALL_BACKENDS", Some("1"))]);
    assert!(require_all_backends());
}

#[test]
fn dist_preparation_removes_stale_backend_artifacts_and_staging_dir() {
    let temp = TempDir::new("target-node-prepare");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let dist = temp.create_dir("dist");
    temp.write("dist/sipp_node_cpu.win32-x64-msvc.node", "");
    temp.write("dist/keep.node", "");
    temp.write("dist/sipp_node_cpu.txt", "");
    temp.write(".build/tmp/node/stale/file.txt", "");
    let sh = xshell::Shell::new().unwrap();

    prepare_dist_dir(&sh, &ctx, &dist).unwrap();

    assert!(!dist.join("sipp_node_cpu.win32-x64-msvc.node").exists());
    assert!(dist.join("keep.node").exists());
    assert!(dist.join("sipp_node_cpu.txt").exists());
    assert!(!ctx.tmp_dir().join("node").exists());
}
