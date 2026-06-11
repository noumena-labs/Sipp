//! Tests the `toolchains::cuda` module in `xtask`.
//!
//! Covers CUDA discovery success and failure branches, CUDA architecture list
//! resolution with serialized fake environment variables and fixture nvcc
//! paths, as well as config file persistence and compute-capability parsing.

use crate::test_support::{EnvGuard, TempDir};
use crate::utils::BuildContext;

use super::{
    compute_cap_to_arch, cuda_architectures, cuda_architectures_with_source, cuda_config_path,
    read_persisted_architectures, setup_cuda, write_persisted_architectures, ArchSource,
    DEFAULT_CUDA_ARCHITECTURES,
};

// ---------------------------------------------------------------------------
// CUDA toolkit discovery
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Architecture list resolution (env var)
// ---------------------------------------------------------------------------

#[test]
fn cuda_architectures_prefers_trimmed_environment_override() {
    let temp = TempDir::new("cuda-arch-env");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let _env = EnvGuard::new(&[("COGENTLM_CUDA_ARCHITECTURES", Some(" 80;90 "))]);

    let (arches, source) = cuda_architectures_with_source(&ctx);
    assert_eq!(arches, "80;90");
    assert!(matches!(source, ArchSource::EnvVar));
}

#[test]
fn cuda_architectures_defaults_when_unset() {
    let temp = TempDir::new("cuda-arch-default");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let _env = EnvGuard::new(&[("COGENTLM_CUDA_ARCHITECTURES", None)]);

    assert_eq!(cuda_architectures(&ctx), DEFAULT_CUDA_ARCHITECTURES);
}

#[test]
fn cuda_architectures_defaults_when_blank() {
    let temp = TempDir::new("cuda-arch-blank");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let _env = EnvGuard::new(&[("COGENTLM_CUDA_ARCHITECTURES", Some("   "))]);

    assert_eq!(cuda_architectures(&ctx), DEFAULT_CUDA_ARCHITECTURES);
}

// ---------------------------------------------------------------------------
// Architecture list resolution (config file persistence)
// ---------------------------------------------------------------------------

#[test]
fn cuda_architectures_falls_back_to_config_file() {
    let temp = TempDir::new("cuda-arch-config");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    // Write a config file
    write_persisted_architectures(&ctx, "50-real;60-real").unwrap();
    let config_path = cuda_config_path(&ctx);
    assert!(config_path.exists());

    let (arches, source) = cuda_architectures_with_source(&ctx);
    assert_eq!(arches, "50-real;60-real");
    assert!(matches!(source, ArchSource::ConfigFile(_)));
}

#[test]
fn cuda_architectures_env_var_beats_config_file() {
    let temp = TempDir::new("cuda-arch-env-beats-config");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    write_persisted_architectures(&ctx, "50-real").unwrap();
    let _env = EnvGuard::new(&[("COGENTLM_CUDA_ARCHITECTURES", Some("99-real"))]);

    let (arches, source) = cuda_architectures_with_source(&ctx);
    assert_eq!(arches, "99-real");
    assert!(matches!(source, ArchSource::EnvVar));
}

#[test]
fn cuda_architecture_default_beats_empty_config_file() {
    let temp = TempDir::new("cuda-arch-empty-config");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    // Write an empty config file (triggers read, returns None, falls through)
    let config_path = cuda_config_path(&ctx);
    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&config_path, "   ").unwrap();

    let (arches, source) = cuda_architectures_with_source(&ctx);
    assert_eq!(arches, DEFAULT_CUDA_ARCHITECTURES);
    assert!(matches!(source, ArchSource::Default));
}

#[test]
fn read_write_persisted_architectures_round_trips() {
    let temp = TempDir::new("cuda-arch-roundtrip");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    assert!(read_persisted_architectures(&ctx).is_none());

    write_persisted_architectures(&ctx, "75-real;80-real").unwrap();
    assert_eq!(
        read_persisted_architectures(&ctx).as_deref(),
        Some("75-real;80-real")
    );

    // Overwrite
    write_persisted_architectures(&ctx, "90-virtual").unwrap();
    assert_eq!(
        read_persisted_architectures(&ctx).as_deref(),
        Some("90-virtual")
    );
}

// ---------------------------------------------------------------------------
// compute_cap_to_arch
// ---------------------------------------------------------------------------

#[test]
fn compute_cap_to_arch_converts_dot_format() {
    assert_eq!(compute_cap_to_arch("8.9").as_deref(), Some("89-real"));
    assert_eq!(compute_cap_to_arch("7.5").as_deref(), Some("75-real"));
    assert_eq!(compute_cap_to_arch("9.0").as_deref(), Some("90-real"));
}

#[test]
fn compute_cap_to_arch_handles_three_digit() {
    assert_eq!(compute_cap_to_arch("12.0").as_deref(), Some("120-real"));
    assert_eq!(compute_cap_to_arch("12.1").as_deref(), Some("121-real"));
}

#[test]
fn compute_cap_to_arch_returns_none_for_empty_input() {
    assert!(compute_cap_to_arch("").is_none());
}

#[test]
fn compute_cap_to_arch_returns_none_for_non_numeric() {
    assert!(compute_cap_to_arch("abc").is_none());
}

// ---------------------------------------------------------------------------
// ArchSource label
// ---------------------------------------------------------------------------

#[test]
fn arch_source_labels_are_stable() {
    assert_eq!(ArchSource::EnvVar.label(), "env var");
    assert_eq!(ArchSource::Default.label(), "default");
    // ConfigFile label doesn't depend on the path
    assert_eq!(ArchSource::ConfigFile("x".into()).label(), "config file");
}

#[test]
fn arch_source_config_path_returns_path_only_for_config_file_variant() {
    assert!(ArchSource::EnvVar.config_path().is_none());
    assert!(ArchSource::Default.config_path().is_none());
    let path = std::path::PathBuf::from("/some/path");
    assert_eq!(
        ArchSource::ConfigFile(path.clone()).config_path(),
        Some(path.as_path())
    );
}
