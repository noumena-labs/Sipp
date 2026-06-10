//! Tests the `toolchains::cuda` module in `xtask`.
//!
//! Covers CUDA discovery success and failure branches plus CUDA architecture
//! list resolution with serialized fake environment variables and fixture
//! `nvcc` paths instead of probing a real CUDA Toolkit installation.

use crate::test_support::{EnvGuard, TempDir};

use super::{cuda_architectures, setup_cuda, DEFAULT_CUDA_ARCHITECTURES};

#[test]
fn setup_cuda_errors_when_no_cuda_environment_points_at_nvcc() {
    let _env = EnvGuard::new(&[("CUDA_PATH", None), ("CUDA_HOME", None)]);

    let error = setup_cuda().unwrap_err();

    assert!(format!("{error:#}").contains("Missing NVIDIA CUDA Toolkit"));
}

#[test]
fn setup_cuda_returns_fake_root_when_nvcc_exists() {
    let temp = TempDir::new("cuda-present");
    let nvcc = if cfg!(windows) {
        "bin/nvcc.exe"
    } else {
        "bin/nvcc"
    };
    temp.write(nvcc, "");
    let root = temp.path().display().to_string();
    let _env = EnvGuard::new(&[("CUDA_PATH", Some(&root)), ("CUDA_HOME", None)]);

    assert_eq!(setup_cuda().unwrap(), temp.path());
}

#[test]
fn cuda_architectures_prefers_trimmed_environment_override() {
    let _env = EnvGuard::new(&[("COGENTLM_CUDA_ARCHITECTURES", Some(" 80;90 "))]);

    assert_eq!(cuda_architectures(), "80;90");
}

#[test]
fn cuda_architectures_defaults_when_unset() {
    let _env = EnvGuard::new(&[("COGENTLM_CUDA_ARCHITECTURES", None)]);

    assert_eq!(cuda_architectures(), DEFAULT_CUDA_ARCHITECTURES);
}

#[test]
fn cuda_architectures_defaults_when_blank() {
    let _env = EnvGuard::new(&[("COGENTLM_CUDA_ARCHITECTURES", Some("   "))]);

    assert_eq!(cuda_architectures(), DEFAULT_CUDA_ARCHITECTURES);
}
