//! Tests the `toolchains::env` module in `xtask`.
//!
//! Covers deterministic platform path-separator selection and native command
//! environment setup without downloading external toolchains.

#[cfg(target_os = "macos")]
use crate::cli::Backend;
use crate::test_support::TempDir;
#[cfg(target_os = "macos")]
use crate::toolchains::vulkan::VulkanLayout;
use crate::utils::BuildContext;
#[cfg(target_os = "macos")]
use std::fs;
use xshell::cmd;

use super::{apply_toolchains, macos_deployment_target, path_separator, prepend_paths};

#[test]
fn path_separator_matches_host_platform() {
    assert_eq!(path_separator(), if cfg!(windows) { ";" } else { ":" });
}

#[test]
fn prepended_paths_preserve_order_and_existing_entries() {
    let paths = vec!["first".to_string(), "second".to_string()];
    let separator = path_separator();

    assert_eq!(prepend_paths(&paths, ""), format!("first{separator}second"));
    assert_eq!(
        prepend_paths(&paths, "existing"),
        format!("first{separator}second{separator}existing")
    );
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

#[cfg(target_os = "macos")]
#[test]
fn macos_vulkan_env_uses_the_managed_icd() {
    let temp = TempDir::new("env-macos-vulkan");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let layout = VulkanLayout::current(&ctx);
    fs::create_dir_all(layout.glslc.parent().unwrap()).unwrap();
    fs::write(&layout.glslc, b"").unwrap();
    let sh = xshell::Shell::new().unwrap();

    let output = apply_toolchains(&sh, &ctx, cmd!(sh, "/usr/bin/env"), Some(&Backend::Vulkan))
        .unwrap()
        .read()
        .unwrap();

    let icd = layout
        .sdk_dir
        .join("share/vulkan/icd.d/MoltenVK_icd.json")
        .display()
        .to_string();
    let configured_icd = output
        .lines()
        .find_map(|line| line.strip_prefix("VK_ICD_FILENAMES="))
        .unwrap();

    assert_eq!(configured_icd, icd);
}
