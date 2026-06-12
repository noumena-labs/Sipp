//! Tests the `build_support::targets` module in `sipp-sys`.
//!
//! Covers target classification across deterministic OS/triple inputs without
//! requiring host-specific native toolchains.

use super::*;

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
