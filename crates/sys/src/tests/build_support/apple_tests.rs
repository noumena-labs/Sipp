//! Tests Apple native build configuration.
//!
//! Covers explicit Rust-target mappings for macOS, iOS devices, and iOS
//! simulators without invoking CMake or requiring Apple SDKs.

use super::{architecture, default_macos_target, ios_sdk};

#[test]
fn apple_targets_map_to_cmake_architectures() {
    assert_eq!(architecture("aarch64-apple-darwin"), "arm64");
    assert_eq!(architecture("x86_64-apple-darwin"), "x86_64");
    assert_eq!(architecture("aarch64-apple-ios"), "arm64");
    assert_eq!(architecture("aarch64-apple-ios-sim"), "arm64");
    assert_eq!(architecture("x86_64-apple-ios"), "x86_64");
    assert_eq!(default_macos_target("aarch64-apple-darwin"), "11.0");
    assert_eq!(default_macos_target("x86_64-apple-darwin"), "10.15");
}

#[test]
fn ios_targets_map_to_device_and_simulator_sdks() {
    assert_eq!(ios_sdk("aarch64-apple-ios"), "iphoneos");
    assert_eq!(ios_sdk("aarch64-apple-ios-sim"), "iphonesimulator");
    assert_eq!(ios_sdk("x86_64-apple-ios"), "iphonesimulator");
}
