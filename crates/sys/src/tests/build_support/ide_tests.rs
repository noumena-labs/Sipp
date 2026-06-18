//! Tests the `build_support::ide` module in `sipp-sys`.
//!
//! Covers compile database copying and fallback clangd flag generation using
//! fake filesystem fixtures, without running CMake or native compilers.

use std::fs;

use crate::build_support::context::{BuildContext, BuildEnv, FeatureFlags};
use crate::build_support::targets::TargetKind;
use crate::build_support_test_common::TempDir;

use super::*;

fn test_context(temp: &TempDir) -> BuildContext {
    let manifest_dir = temp.join("workspace/crates/sys");
    let llama_dir = temp.join("workspace/crates/sys/llama.cpp");
    fs::create_dir_all(&manifest_dir).expect("manifest dir");
    BuildContext {
        manifest_dir,
        llama_dir,
        target: "x86_64-unknown-linux-gnu".to_owned(),
        target_kind: TargetKind::Unix,
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
            static_cxx_runtime_lib_dir: None,
        },
    }
}

#[test]
fn setup_clangd_autocomplete_copies_compile_database_and_removes_flags() {
    let temp = TempDir::new("ide-db");
    let context = test_context(&temp);
    let dst = temp.join("cmake-out");
    let build_dir = dst.join("build");
    fs::create_dir_all(&build_dir).expect("cmake build dir");
    fs::write(
        build_dir.join("compile_commands.json"),
        "[{\"directory\":\"x\"}]",
    )
    .expect("compile database");

    let ide_dir = context.workspace_build_dir().join("ide").join("sys");
    fs::create_dir_all(&ide_dir).expect("ide dir");
    fs::write(ide_dir.join("compile_flags.txt"), "stale").expect("stale flags");

    setup_clangd_autocomplete(&context, &dst);

    assert_eq!(
        fs::read_to_string(ide_dir.join("compile_commands.json")).expect("copied db"),
        "[{\"directory\":\"x\"}]"
    );
    assert!(!ide_dir.join("compile_flags.txt").exists());
}

#[test]
fn setup_clangd_autocomplete_writes_fallback_flags_and_removes_stale_database() {
    let temp = TempDir::new("ide-flags");
    let context = test_context(&temp);
    let dst = temp.join("cmake-out");
    fs::create_dir_all(dst.join("build")).expect("cmake build dir");

    let ide_dir = context.workspace_build_dir().join("ide").join("sys");
    fs::create_dir_all(&ide_dir).expect("ide dir");
    fs::write(ide_dir.join("compile_commands.json"), "stale").expect("stale db");

    setup_clangd_autocomplete(&context, &dst);

    let flags = fs::read_to_string(ide_dir.join("compile_flags.txt")).expect("fallback flags");
    assert!(flags.contains("-xc++"));
    assert!(flags.contains("-std=c++17"));
    assert!(flags.contains(&format!(
        "-I{}",
        context.manifest_dir.join("native/cxx_bridge").display()
    )));
    assert!(flags.contains(&format!("-I{}", context.llama_dir.join("vendor").display())));
    assert!(flags.contains("-DGGML_USE_OPENMP=OFF"));
    assert!(!ide_dir.join("compile_commands.json").exists());
}

#[test]
fn setup_clangd_autocomplete_preserves_existing_source_flags_as_noop() {
    let temp = TempDir::new("ide-source-flags");
    let context = test_context(&temp);
    let dst = temp.join("cmake-out");
    let build_dir = dst.join("build");
    fs::create_dir_all(&build_dir).expect("cmake build dir");
    fs::write(build_dir.join("compile_flags.txt"), "source flags").expect("source flags");

    let ide_dir = context.workspace_build_dir().join("ide").join("sys");
    fs::create_dir_all(&ide_dir).expect("ide dir");
    fs::write(ide_dir.join("compile_flags.txt"), "existing flags").expect("existing flags");
    fs::write(ide_dir.join("compile_commands.json"), "existing db").expect("existing db");

    setup_clangd_autocomplete(&context, &dst);

    assert_eq!(
        fs::read_to_string(ide_dir.join("compile_flags.txt")).expect("ide flags"),
        "existing flags"
    );
    assert_eq!(
        fs::read_to_string(ide_dir.join("compile_commands.json")).expect("ide db"),
        "existing db"
    );
}
