//! Tests the `build_support::link` module in `sipp-sys`.
//!
//! Covers deterministic library search and archive/shared-library detection
//! with fake filesystem fixtures, without invoking the linker.

use std::fs;

use crate::build_support::targets::TargetKind;
use crate::build_support_test_common::TempDir;

use super::*;

#[test]
fn library_search_dirs_collects_existing_candidates_in_order() {
    let temp = TempDir::new("link-search-order");
    let dst = temp.path();
    let lib = temp.join("lib");
    let lib_release = lib.join("Release");
    let bin_debug = temp.join("bin/Debug");
    fs::create_dir_all(&lib_release).expect("lib release dir");
    fs::create_dir_all(&bin_debug).expect("bin debug dir");

    assert_eq!(
        library_search_dirs(dst, &lib),
        vec![lib, lib_release, temp.join("bin"), bin_debug]
    );
}

#[test]
fn library_search_dirs_suppresses_duplicate_candidates() {
    let temp = TempDir::new("link-search-duplicates");
    let bin = temp.join("bin");
    fs::create_dir_all(&bin).expect("bin dir");

    assert_eq!(library_search_dirs(temp.path(), &bin), vec![bin]);
}

#[test]
fn static_library_exists_recognizes_supported_archive_names() {
    let temp = TempDir::new("link-static");
    let search_dir = temp.join("search");
    fs::create_dir_all(&search_dir).expect("search dir");
    fs::write(search_dir.join("libalpha.a"), b"").expect("alpha archive");
    fs::write(search_dir.join("beta.lib"), b"").expect("beta archive");
    fs::write(search_dir.join("libgamma.lib"), b"").expect("gamma archive");
    fs::write(search_dir.join("delta.a"), b"").expect("delta archive");
    let dirs = vec![search_dir];

    for lib in ["alpha", "beta", "gamma", "delta"] {
        assert!(static_library_exists(&dirs, lib), "{lib}");
    }
    assert!(!static_library_exists(&dirs, "missing"));
}

#[test]
fn dynamic_library_exists_recognizes_windows_import_libraries() {
    let temp = TempDir::new("link-dynamic-windows");
    let search_dir = temp.join("search");
    fs::create_dir_all(&search_dir).expect("search dir");
    fs::write(search_dir.join("alpha.lib"), b"").expect("alpha import lib");
    fs::write(search_dir.join("libbeta.dll.a"), b"").expect("beta import lib");
    let dirs = vec![search_dir];

    assert!(dynamic_library_exists(TargetKind::Windows, &dirs, "alpha"));
    assert!(dynamic_library_exists(TargetKind::Windows, &dirs, "beta"));
    assert!(!dynamic_library_exists(
        TargetKind::Windows,
        &dirs,
        "missing"
    ));
}

#[test]
fn dynamic_library_exists_recognizes_unix_and_macos_shared_libraries() {
    let temp = TempDir::new("link-dynamic-unix");
    let search_dir = temp.join("search");
    fs::create_dir_all(&search_dir).expect("search dir");
    fs::write(search_dir.join("libalpha.so"), b"").expect("alpha so");
    fs::write(search_dir.join("libbeta.dylib"), b"").expect("beta dylib");
    fs::write(search_dir.join("libgamma.tbd"), b"").expect("gamma tbd");
    let dirs = vec![search_dir];

    for lib in ["alpha", "beta", "gamma"] {
        assert!(
            dynamic_library_exists(TargetKind::Unix, &dirs, lib),
            "{lib}"
        );
        assert!(
            dynamic_library_exists(TargetKind::Macos, &dirs, lib),
            "{lib}"
        );
    }
    assert!(!dynamic_library_exists(TargetKind::Unix, &dirs, "missing"));
}
