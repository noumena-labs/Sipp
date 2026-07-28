//! Tests the `toolchains::vulkan` module in `xtask`.
//!
//! Covers deterministic SDK, archive, and macOS installer paths for every
//! supported host without downloading or executing the Vulkan SDK.

use std::path::Path;

use crate::test_support::TempDir;
use crate::utils::BuildContext;
use xshell::Shell;

use super::{macos_installer_path, setup_vulkan, VulkanHost, VulkanLayout, VULKAN_VERSION};

#[test]
fn sdk_layouts_are_versioned_and_platform_specific() {
    let cache = Path::new("toolchain").join("vulkan");
    let version = cache.join(VULKAN_VERSION);

    let windows = VulkanLayout::new(&cache, VulkanHost::Windows);
    assert_eq!(windows.install_dir, version);
    assert_eq!(windows.sdk_dir, version);
    assert_eq!(windows.bin_dir, version.join("Bin"));
    assert_eq!(windows.glslc, version.join("Bin/glslc.exe"));

    let macos = VulkanLayout::new(&cache, VulkanHost::Macos);
    assert_eq!(macos.install_dir, version);
    assert_eq!(macos.sdk_dir, version.join("macOS"));
    assert_eq!(macos.bin_dir, version.join("macOS/bin"));
    assert_eq!(macos.glslc, version.join("macOS/bin/glslc"));

    let linux = VulkanLayout::new(&cache, VulkanHost::Linux);
    assert_eq!(linux.install_dir, version);
    assert_eq!(linux.sdk_dir, version.join("x86_64"));
    assert_eq!(linux.bin_dir, version.join("x86_64/bin"));
    assert_eq!(linux.glslc, version.join("x86_64/bin/glslc"));
}

#[test]
fn download_coordinates_match_lunarg_packages() {
    assert_eq!(VulkanHost::Windows.download_path(), "windows");
    assert_eq!(
        VulkanHost::Windows.archive_name(),
        format!("vulkansdk-windows-X64-{VULKAN_VERSION}.exe")
    );
    assert_eq!(VulkanHost::Macos.download_path(), "mac");
    assert_eq!(
        VulkanHost::Macos.archive_name(),
        format!("vulkansdk-macos-{VULKAN_VERSION}.zip")
    );
    assert_eq!(VulkanHost::Linux.download_path(), "linux");
    assert_eq!(
        VulkanHost::Linux.archive_name(),
        format!("vulkansdk-linux-x86_64-{VULKAN_VERSION}.tar.xz")
    );
}

#[test]
fn macos_installer_path_targets_the_extracted_app_executable() {
    let staging = Path::new("tmp").join("vulkan");
    let name = format!("vulkansdk-macOS-{VULKAN_VERSION}");

    assert_eq!(
        macos_installer_path(&staging),
        staging
            .join(format!("{name}.app"))
            .join("Contents/MacOS")
            .join(name)
    );
}

#[test]
fn existing_versioned_compiler_skips_installation() {
    let temp = TempDir::new("vulkan-installed");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let expected = VulkanLayout::current(&ctx);
    temp.write(&expected.glslc, "");
    let sh = Shell::new().expect("create shell");

    let installed = setup_vulkan(&sh, &ctx).expect("reuse Vulkan SDK");

    assert_eq!(installed.sdk_dir, expected.sdk_dir);
    assert_eq!(installed.bin_dir, expected.bin_dir);
    assert_eq!(installed.glslc, expected.glslc);
}
