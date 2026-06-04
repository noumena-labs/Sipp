//! Tests the `targets::python` module in `xtask`.
//!
//! Covers Python project paths, backend expansion, wheel/native extension
//! discovery, and artifact cleanup with fake directories instead of running uv
//! or Maturin.

use crate::cli::Backend;
use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{
    backend_binary_dir, backend_label, backends_to_build, find_native_extension,
    find_wheel_artifact, is_python_extension, prepare_backend_binary_dir, prepare_dist_dir,
    python_package_dir, python_project_dir,
};

#[test]
fn python_paths_are_rooted_under_bindings_python() {
    let temp = TempDir::new("target-python-paths");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    assert_eq!(python_project_dir(&ctx), temp.join("bindings/python"));
    assert_eq!(
        python_package_dir(&ctx),
        temp.join("bindings/python/python/cogentlm")
    );
    assert_eq!(
        backend_binary_dir(&ctx),
        temp.join("bindings/python/python/cogentlm/binaries")
    );
}

#[test]
fn backend_expansion_and_labels_are_stable() {
    if cfg!(target_os = "macos") {
        assert_eq!(backends_to_build(), vec![Backend::Cpu, Backend::Metal]);
    } else {
        assert_eq!(
            backends_to_build(),
            vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda]
        );
    }
    assert_eq!(backend_label(None), "cpu (default)");
    assert_eq!(backend_label(Some(&Backend::Vulkan)), "vulkan");
}

#[test]
fn wheel_and_native_extension_discovery_match_packaging_conventions() {
    let temp = TempDir::new("target-python-discovery");
    let wheel = temp.write("dist/cogentlm-0.1.0-py3-none-any.whl", "");
    temp.write("dist/readme.txt", "");
    let extension = if cfg!(windows) {
        temp.write("extracted/cogentlm/_native.pyd", "")
    } else {
        temp.write("extracted/cogentlm/_native.so", "")
    };
    temp.write("extracted/cogentlm/other.py", "");

    assert_eq!(
        find_wheel_artifact(&temp.join("dist")).unwrap(),
        Some(wheel)
    );
    assert_eq!(
        find_native_extension(&temp.join("extracted")).unwrap(),
        Some(extension)
    );
    assert!(is_python_extension(std::path::Path::new("_native.pyd")));
    assert!(is_python_extension(std::path::Path::new("_native.so")));
    assert!(!is_python_extension(std::path::Path::new("_native.dll")));
}

#[test]
fn backend_binary_preparation_preserves_gitkeep_only() {
    let temp = TempDir::new("target-python-binaries");
    let dir = temp.create_dir("binaries");
    temp.write("binaries/.gitkeep", "");
    temp.write("binaries/_native_cpu.pyd", "");
    temp.write("binaries/nested/file.txt", "");
    let sh = xshell::Shell::new().unwrap();

    prepare_backend_binary_dir(&sh, &dir).unwrap();

    assert!(dir.join(".gitkeep").exists());
    assert!(!dir.join("_native_cpu.pyd").exists());
    assert!(!dir.join("nested").exists());
}

#[test]
fn dist_preparation_removes_only_cogentlm_wheels() {
    let temp = TempDir::new("target-python-dist");
    let dist = temp.create_dir("dist");
    temp.write("dist/cogentlm-0.1.0.whl", "");
    temp.write("dist/other-0.1.0.whl", "");
    temp.write("dist/cogentlm-0.1.0.txt", "");
    let sh = xshell::Shell::new().unwrap();

    prepare_dist_dir(&sh, &dist).unwrap();

    assert!(!dist.join("cogentlm-0.1.0.whl").exists());
    assert!(dist.join("other-0.1.0.whl").exists());
    assert!(dist.join("cogentlm-0.1.0.txt").exists());
}
