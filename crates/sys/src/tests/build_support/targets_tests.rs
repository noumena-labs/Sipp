//! Tests the `build_support::targets` module in `sipp-sys`.
//!
//! Covers target classification across deterministic OS/triple inputs without
//! requiring host-specific native toolchains.

use super::*;
use crate::build_support::context::{BuildContext, BuildEnv, FeatureFlags};

fn target_context(target: &str) -> BuildContext {
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
