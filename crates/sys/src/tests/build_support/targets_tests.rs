//! Tests the `build_support::targets` module in `sipp-sys`.
//!
//! Covers target classification across deterministic OS/triple inputs without
//! requiring host-specific native toolchains.

use super::*;
use crate::build_support::context::{BuildContext, BuildEnv, FeatureFlags};
use crate::build_support_test_common::TempDir;
use std::fs;
use std::path::PathBuf;

fn target_context(target: &str) -> BuildContext {
    target_context_with_static_cxx_lib_dir(target, None)
}

fn target_context_with_static_cxx_lib_dir(
    target: &str,
    static_cxx_runtime_lib_dir: Option<PathBuf>,
) -> BuildContext {
    BuildContext {
        manifest_dir: "crates/sys".into(),
        llama_dir: "crates/sys/llama.cpp".into(),
        target: target.to_owned(),
        target_kind: TargetKind::Macos,
        host_is_windows: false,
        features: FeatureFlags {
            backend_dl: false,
            cuda: false,
            metal: false,
            vulkan: false,
            openmp: false,
            pthreads: false,
        },
        env_vars: BuildEnv {
            cuda_path: None,
            cuda_architectures: None,
            vulkan_sdk: None,
            cmake_out_dir: None,
            static_cxx_runtime_lib_dir,
        },
    }
}

#[test]
fn classify_prefers_emscripten_os_or_triple() {
    assert_eq!(
        classify("emscripten", "wasm32-unknown"),
        TargetKind::Emscripten
    );
    assert_eq!(
        classify("unknown", "wasm32-unknown-emscripten"),
        TargetKind::Emscripten
    );
}

#[test]
fn classify_maps_known_host_operating_systems() {
    assert_eq!(
        classify("windows", "x86_64-pc-windows-msvc"),
        TargetKind::Windows
    );
    assert_eq!(classify("macos", "aarch64-apple-darwin"), TargetKind::Macos);
    assert_eq!(
        classify("linux", "x86_64-unknown-linux-gnu"),
        TargetKind::Unix
    );
    assert_eq!(classify("", ""), TargetKind::Unix);
}

#[test]
fn target_kind_reports_only_emscripten_as_emscripten() {
    assert!(TargetKind::Emscripten.is_emscripten());
    assert!(!TargetKind::Windows.is_emscripten());
    assert!(!TargetKind::Macos.is_emscripten());
    assert!(!TargetKind::Unix.is_emscripten());
}

#[test]
fn macos_deployment_target_supports_filesystem_floor() {
    assert_eq!(
        super::macos::deployment_target(&target_context("x86_64-apple-darwin")),
        "10.15"
    );
    assert_eq!(
        super::macos::deployment_target(&target_context("aarch64-apple-darwin")),
        "11.0"
    );
}

#[test]
fn unix_cuda_flags_compile_position_independent_objects() {
    assert!(super::unix::cuda_cmake_flags().contains("-fPIC"));
}

#[test]
fn linux_links_dynamic_cpp_runtime_by_default() {
    assert_eq!(
        super::unix::stdcpp_link_kind(&target_context("x86_64-unknown-linux-gnu")),
        "dylib=stdc++"
    );
}

#[test]
fn linux_links_static_cpp_runtime_for_portable_artifacts() {
    let temp = TempDir::new("static-stdcpp");
    let lib_dir = temp.join("gcc");
    fs::create_dir_all(&lib_dir).expect("lib dir");
    fs::write(lib_dir.join("libstdc++.a"), "").expect("archive");

    let linux_context =
        target_context_with_static_cxx_lib_dir("x86_64-unknown-linux-gnu", Some(lib_dir.clone()));

    assert_eq!(
        super::unix::stdcpp_link_kind(&linux_context),
        "static=stdc++"
    );
    assert_eq!(super::unix::static_stdcpp_lib_dir(&linux_context), lib_dir);
    assert_eq!(
        super::unix::stdcpp_link_kind(&target_context_with_static_cxx_lib_dir(
            "x86_64-unknown-freebsd",
            Some(lib_dir),
        )),
        "dylib=stdc++"
    );
}
