//! Tests the `build_support::context` module in `cogentlm-sys`.
//!
//! Covers environment-derived build settings, backend tag selection, output
//! paths, and vendored llama.cpp validation with deterministic fake paths.

use std::fs;
use std::path::PathBuf;

use crate::build_support::targets::TargetKind;
use crate::build_support_test_common::{EnvGuard, TempDir};

use super::*;

const CONTEXT_ENV_VARS: &[(&str, Option<&str>)] = &[
    ("CARGO_MANIFEST_DIR", None),
    ("TARGET", None),
    ("CARGO_CFG_TARGET_OS", None),
    ("CARGO_FEATURE_BACKEND_DL", None),
    ("CARGO_FEATURE_CUDA", None),
    ("CARGO_FEATURE_METAL", None),
    ("CARGO_FEATURE_VULKAN", None),
    ("CARGO_FEATURE_OPENMP", None),
    ("CARGO_FEATURE_PTHREADS", None),
    ("CARGO_FEATURE_PTHREAD", None),
    ("CUDA_PATH", None),
    ("CUDA_HOME", None),
    ("COGENTLM_CUDA_ARCHITECTURES", None),
    ("VULKAN_SDK", None),
    ("COGENTLM_SYS_CMAKE_OUT_DIR", None),
];

fn flags(backend_dl: bool, cuda: bool, metal: bool, vulkan: bool, openmp: bool) -> FeatureFlags {
    FeatureFlags {
        backend_dl,
        cuda,
        metal,
        vulkan,
        openmp,
        pthreads: false,
    }
}

fn context_for(manifest_dir: PathBuf, target: &str, features: FeatureFlags) -> BuildContext {
    BuildContext {
        manifest_dir,
        llama_dir: PathBuf::from("llama.cpp"),
        target: target.to_owned(),
        target_kind: TargetKind::Unix,
        host_is_windows: false,
        features,
        env_vars: BuildEnv {
            cuda_path: None,
            cuda_architectures: None,
            vulkan_sdk: None,
            cmake_out_dir: None,
        },
    }
}

#[test]
fn feature_flags_from_env_reads_all_supported_flags() {
    let _env = EnvGuard::new(&[
        ("CARGO_FEATURE_BACKEND_DL", Some("1")),
        ("CARGO_FEATURE_CUDA", Some("1")),
        ("CARGO_FEATURE_METAL", Some("1")),
        ("CARGO_FEATURE_VULKAN", Some("1")),
        ("CARGO_FEATURE_OPENMP", Some("1")),
        ("CARGO_FEATURE_PTHREADS", Some("1")),
        ("CARGO_FEATURE_PTHREAD", None),
    ]);

    let flags = FeatureFlags::from_env();
    assert!(flags.backend_dl);
    assert!(flags.cuda);
    assert!(flags.metal);
    assert!(flags.vulkan);
    assert!(flags.openmp);
    assert!(flags.pthreads);
}

#[test]
fn feature_flags_from_env_accepts_singular_pthread_flag() {
    let _env = EnvGuard::new(&[
        ("CARGO_FEATURE_BACKEND_DL", None),
        ("CARGO_FEATURE_CUDA", None),
        ("CARGO_FEATURE_METAL", None),
        ("CARGO_FEATURE_VULKAN", None),
        ("CARGO_FEATURE_OPENMP", None),
        ("CARGO_FEATURE_PTHREADS", None),
        ("CARGO_FEATURE_PTHREAD", Some("1")),
    ]);

    let flags = FeatureFlags::from_env();
    assert!(!flags.backend_dl);
    assert!(!flags.cuda);
    assert!(!flags.metal);
    assert!(!flags.vulkan);
    assert!(!flags.openmp);
    assert!(flags.pthreads);
}

#[test]
fn backend_tag_uses_backend_priority_and_dynamic_prefix() {
    assert_eq!(flags(false, false, false, false, false).backend_tag(), "c");
    assert_eq!(flags(false, false, false, false, true).backend_tag(), "om");
    assert_eq!(flags(false, false, false, true, true).backend_tag(), "vk");
    assert_eq!(flags(false, false, true, true, true).backend_tag(), "mt");
    assert_eq!(flags(false, true, true, true, true).backend_tag(), "cu");
    assert_eq!(
        flags(true, false, false, false, false).backend_tag(),
        "dl-c"
    );
    assert_eq!(flags(true, true, true, true, true).backend_tag(), "dl-cu");
}

#[test]
fn build_env_prefers_cuda_path_and_sanitizes_cmake_output() {
    let _env = EnvGuard::new(&[
        ("CUDA_PATH", Some("cuda-path")),
        ("CUDA_HOME", Some("cuda-home")),
        ("COGENTLM_CUDA_ARCHITECTURES", Some(" 80;90 ")),
        ("VULKAN_SDK", Some("vulkan-sdk")),
        ("COGENTLM_SYS_CMAKE_OUT_DIR", Some(r"\\?\cmake-out")),
    ]);

    let env = BuildEnv::from_env();
    assert_eq!(env.cuda_path, Some(PathBuf::from("cuda-path")));
    assert_eq!(env.cuda_architectures, Some("80;90".to_owned()));
    assert_eq!(env.vulkan_sdk, Some(PathBuf::from("vulkan-sdk")));
    assert_eq!(env.cmake_out_dir, Some(PathBuf::from("cmake-out")));
}

#[test]
fn build_env_falls_back_to_cuda_home() {
    let _env = EnvGuard::new(&[
        ("CUDA_PATH", None),
        ("CUDA_HOME", Some("cuda-home")),
        ("COGENTLM_CUDA_ARCHITECTURES", None),
        ("VULKAN_SDK", None),
        ("COGENTLM_SYS_CMAKE_OUT_DIR", None),
    ]);

    let env = BuildEnv::from_env();
    assert_eq!(env.cuda_path, Some(PathBuf::from("cuda-home")));
    assert_eq!(env.cuda_architectures, None);
    assert_eq!(env.vulkan_sdk, None);
    assert_eq!(env.cmake_out_dir, None);
}

#[test]
fn build_env_treats_blank_cuda_architectures_as_unset() {
    let _env = EnvGuard::new(&[
        ("CUDA_PATH", None),
        ("CUDA_HOME", None),
        ("COGENTLM_CUDA_ARCHITECTURES", Some("   ")),
        ("VULKAN_SDK", None),
        ("COGENTLM_SYS_CMAKE_OUT_DIR", None),
    ]);

    let env = BuildEnv::from_env();
    assert_eq!(env.cuda_architectures, None);
}

#[test]
fn build_context_new_reads_manifest_target_features_and_env() {
    let temp = TempDir::new("context-new");
    let manifest_dir = temp.join("workspace/crates/sys");
    fs::create_dir_all(&manifest_dir).expect("manifest dir");
    let raw_manifest_dir = format!(r"\\?\{}", manifest_dir.display());
    let mut vars = CONTEXT_ENV_VARS.to_vec();
    vars.extend([
        ("CARGO_MANIFEST_DIR", Some(raw_manifest_dir.as_str())),
        ("TARGET", Some("wasm32-unknown-emscripten")),
        ("CARGO_CFG_TARGET_OS", Some("unknown")),
        ("CARGO_FEATURE_VULKAN", Some("1")),
        ("VULKAN_SDK", Some("sdk")),
    ]);
    let _env = EnvGuard::new(&vars);

    let context = BuildContext::new();
    assert_eq!(context.manifest_dir, manifest_dir);
    assert_eq!(
        context.llama_dir,
        context.manifest_dir.join("../../third_party/llama.cpp")
    );
    assert_eq!(context.target, "wasm32-unknown-emscripten");
    assert_eq!(context.target_kind, TargetKind::Emscripten);
    assert!(context.features.vulkan);
    assert_eq!(context.env_vars.vulkan_sdk, Some(PathBuf::from("sdk")));
}

#[test]
fn workspace_and_default_cmake_paths_use_target_and_backend_tags() {
    let temp = TempDir::new("context-paths");
    let manifest_dir = temp.join("workspace/crates/sys");
    let context = context_for(
        manifest_dir.clone(),
        r"wasm32\unknown:emscripten",
        flags(true, false, false, true, false),
    );

    let workspace_build_dir = manifest_dir.join("../../.build");
    assert_eq!(context.workspace_build_dir(), workspace_build_dir);
    assert_eq!(
        context.default_cmake_out_dir(),
        manifest_dir
            .join("../../.build")
            .join("cmake")
            .join("sys")
            .join("wasm32_unknown_emscripten")
            .join("dl-vk")
    );
}

#[test]
fn validate_llama_dir_accepts_expected_header() {
    let temp = TempDir::new("context-llama-ok");
    let llama_dir = temp.join("third_party/llama.cpp");
    fs::create_dir_all(llama_dir.join("include")).expect("llama include dir");
    fs::write(llama_dir.join("include/llama.h"), b"").expect("llama header");
    let mut context = context_for(
        temp.join("workspace/crates/sys"),
        "",
        flags(false, false, false, false, false),
    );
    context.llama_dir = llama_dir;

    context.validate_llama_dir();
}

#[test]
fn validate_llama_dir_panics_for_missing_header() {
    let temp = TempDir::new("context-llama-missing");
    let mut context = context_for(
        temp.join("workspace/crates/sys"),
        "",
        flags(false, false, false, false, false),
    );
    context.llama_dir = temp.join("third_party/llama.cpp");

    let panic = std::panic::catch_unwind(|| context.validate_llama_dir())
        .expect_err("missing llama.h should panic");
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(message.contains("vendored llama.cpp directory must exist"));
}
