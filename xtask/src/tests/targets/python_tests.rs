//! Tests the `targets::python` module in `xtask`.
//!
//! Covers Python project paths, backend expansion, wheel discovery, generated
//! package names, and artifact cleanup with fake directories instead of running
//! uv or Maturin.

use crate::cli::Backend;
use crate::test_support::{EnvGuard, TempDir};
use crate::utils::BuildContext;

use super::{
    backend_auditwheel_mode, backend_distribution_name, backend_label, backend_package_project_dir,
    backends_to_build, find_wheel_artifact_for_distribution, prepare_dist_dir, python_project_dir,
    require_all_backends, wheel_matches_distribution,
};

#[test]
fn python_paths_are_rooted_under_lib_python() {
    let temp = TempDir::new("target-python-paths");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    assert_eq!(python_project_dir(&ctx), temp.join("lib/python"));
    assert_eq!(
        backend_package_project_dir(&ctx, &Backend::Cuda),
        temp.join("lib/python/backends/cuda")
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
fn strict_backend_policy_requires_exact_env_value() {
    let _env = EnvGuard::new(&[("SIPP_REQUIRE_ALL_BACKENDS", None)]);
    assert!(!require_all_backends());
    drop(_env);

    let _env = EnvGuard::new(&[("SIPP_REQUIRE_ALL_BACKENDS", Some("true"))]);
    assert!(!require_all_backends());
    drop(_env);

    let _env = EnvGuard::new(&[("SIPP_REQUIRE_ALL_BACKENDS", Some("1"))]);
    assert!(require_all_backends());
}

#[test]
fn backend_package_names_match_python_packaging_conventions() {
    assert_eq!(
        backend_distribution_name(&Backend::Cuda),
        "sipp-py-backend-cuda"
    );
}

#[test]
fn linux_gpu_backend_wheels_warn_about_external_driver_libraries() {
    if cfg!(target_os = "linux") {
        assert_eq!(backend_auditwheel_mode(&Backend::Cuda), Some("warn"));
        assert_eq!(backend_auditwheel_mode(&Backend::Vulkan), Some("warn"));
    } else {
        assert_eq!(backend_auditwheel_mode(&Backend::Cuda), None);
        assert_eq!(backend_auditwheel_mode(&Backend::Vulkan), None);
    }
    assert_eq!(backend_auditwheel_mode(&Backend::Cpu), None);
}

#[test]
fn wheel_discovery_matches_normalized_distribution_names() {
    let temp = TempDir::new("target-python-discovery");
    let sipp = temp.write("dist/sipp_py-0.1.0-py3-none-any.whl", "");
    let cuda = temp.write(
        "dist/sipp_py_backend_cuda-0.1.0-cp310-abi3-win_amd64.whl",
        "",
    );
    temp.write("dist/readme.txt", "");

    assert_eq!(
        find_wheel_artifact_for_distribution(&temp.join("dist"), "sipp-py").unwrap(),
        Some(sipp)
    );
    assert_eq!(
        find_wheel_artifact_for_distribution(&temp.join("dist"), "sipp-py-backend-cuda").unwrap(),
        Some(cuda.clone())
    );
    assert!(wheel_matches_distribution(&cuda, "sipp-py-backend-cuda"));
    assert!(!wheel_matches_distribution(&cuda, "sipp-py-backend-vulkan"));
}

#[test]
fn dist_preparation_removes_only_sipp_package_wheels() {
    let temp = TempDir::new("target-python-dist");
    let dist = temp.create_dir("dist");
    temp.write("dist/sipp_py-0.1.0.whl", "");
    temp.write("dist/sipp_py_backend_cuda-0.1.0.whl", "");
    temp.write("dist/sipp-0.1.0.whl", "");
    temp.write("dist/sipp_backend_cuda-0.1.0.whl", "");
    temp.write("dist/other-0.1.0.whl", "");
    temp.write("dist/sipp_py-0.1.0.txt", "");
    let sh = xshell::Shell::new().unwrap();

    prepare_dist_dir(&sh, &dist).unwrap();

    assert!(!dist.join("sipp_py-0.1.0.whl").exists());
    assert!(!dist.join("sipp_py_backend_cuda-0.1.0.whl").exists());
    assert!(!dist.join("sipp-0.1.0.whl").exists());
    assert!(!dist.join("sipp_backend_cuda-0.1.0.whl").exists());
    assert!(dist.join("other-0.1.0.whl").exists());
    assert!(dist.join("sipp_py-0.1.0.txt").exists());
}
