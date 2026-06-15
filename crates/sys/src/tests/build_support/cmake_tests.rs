//! Tests the `build_support::cmake` module in `sipp-sys`.
//!
//! Covers deterministic Vulkan loader discovery with fake SDK files, without
//! invoking CMake or native compilers.

use crate::build_support_test_common::TempDir;
use std::path::Path;

use super::*;

#[test]
fn vulkan_loader_prefers_sdk_loader_library() {
    let temp = TempDir::new("vulkan-loader-sdk");
    let sdk = temp.join("sdk");
    let loader = sdk.join("lib/libvulkan.so.1");
    std::fs::create_dir_all(loader.parent().expect("loader parent")).expect("sdk lib dir");
    std::fs::write(&loader, b"").expect("loader file");

    assert_eq!(vulkan_loader_library(&sdk), Some(loader));
}

#[test]
fn vulkan_loader_returns_none_without_sdk_or_system_loader() {
    let temp = TempDir::new("vulkan-loader-missing");

    if Path::new("/usr/lib/x86_64-linux-gnu/libvulkan.so").exists()
        || Path::new("/usr/lib64/libvulkan.so").exists()
        || Path::new("/usr/lib/libvulkan.so").exists()
    {
        return;
    }

    assert_eq!(vulkan_loader_library(&temp.join("sdk")), None);
}
