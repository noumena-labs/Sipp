//! Tests the `toolchains::env` module in `xtask`.
//!
//! Covers deterministic platform path-separator selection and native command
//! environment setup without downloading external toolchains.

use crate::test_support::TempDir;
use crate::utils::BuildContext;
use xshell::cmd;

use super::{apply_toolchains, macos_deployment_target, path_separator};

#[test]
fn path_separator_matches_host_platform() {
    assert_eq!(path_separator(), if cfg!(windows) { ";" } else { ":" });
}

#[test]
fn macos_deployment_target_matches_host_architecture() {
    let expected = if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            Some("11.0")
        } else {
            Some("10.15")
        }
    } else {
        None
    };

    assert_eq!(macos_deployment_target(), expected);
}

#[test]
fn native_toolchain_env_does_not_install_missing_ninja() {
    let temp = TempDir::new("env-no-ninja-install");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let sh = xshell::Shell::new().unwrap();

    let _command = apply_toolchains(&sh, &ctx, cmd!(sh, "true"), None).unwrap();

    assert!(!ctx.ninja_toolchain_dir().exists());
}
