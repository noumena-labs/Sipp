use crate::build_support::targets::{self, TargetKind};
use crate::build_support::util::{path_component, sanitize_path};
use std::env;
use std::path::PathBuf;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../src/tests/build_support/context_tests.rs"]
mod context_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) struct BuildContext {
    pub(crate) manifest_dir: PathBuf,
    pub(crate) llama_dir: PathBuf,
    pub(crate) target: String,
    pub(crate) target_kind: TargetKind,
    pub(crate) host_is_windows: bool,
    pub(crate) features: FeatureFlags,
    pub(crate) env_vars: BuildEnv,
}

impl BuildContext {
    pub(crate) fn new() -> Self {
        let manifest_dir =
            sanitize_path(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
        let llama_dir = manifest_dir.join("llama.cpp");
        let target = env::var("TARGET").unwrap_or_default();
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_kind = targets::classify(&target_os, &target);

        Self {
            manifest_dir,
            llama_dir,
            target,
            target_kind,
            host_is_windows: cfg!(windows),
            features: FeatureFlags::from_env(),
            env_vars: BuildEnv::from_env(),
        }
    }

    pub(crate) fn validate_llama_dir(&self) {
        assert!(
            self.llama_dir.join("include/llama.h").exists(),
            "vendored llama.cpp directory must exist at {:?}",
            self.llama_dir
        );
    }

    pub(crate) fn emit_rerun_triggers(&self) {
        println!("cargo:rerun-if-changed=CMakeLists.txt");
        println!("cargo:rerun-if-changed=cmake/llama_mtmd_sources.cmake");
        println!("cargo:rerun-if-changed=src/bridge.rs");
        println!("cargo:rerun-if-changed=native/llama_shim/sipp_shim.h");
        println!("cargo:rerun-if-changed=native/llama_shim/sipp_shim.cpp");
        println!("cargo:rerun-if-changed=native/cxx_bridge/sipp_cxx.h");
        println!("cargo:rerun-if-changed=native/cxx_bridge/sipp_cxx.cpp");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=VULKAN_SDK");
        println!("cargo:rerun-if-env-changed=SIPP_GLSLC");
        println!("cargo:rerun-if-env-changed=EMSDK");
        println!("cargo:rerun-if-env-changed=SIPP_SYS_CMAKE_OUT_DIR");
        println!("cargo:rerun-if-env-changed=SIPP_CUDA_ARCHITECTURES");
        println!("cargo:rerun-if-env-changed=SIPP_STATIC_CXX_RUNTIME_LIB_DIR");
    }

    pub(crate) fn workspace_build_dir(&self) -> PathBuf {
        self.manifest_dir.join("../../.build")
    }

    pub(crate) fn default_cmake_out_dir(&self) -> PathBuf {
        self.workspace_build_dir()
            .join("cmake")
            .join("sys")
            .join(path_component(&self.target, "host"))
            .join(self.features.backend_tag())
    }
}

pub(crate) struct FeatureFlags {
    pub(crate) backend_dl: bool,
    pub(crate) cuda: bool,
    pub(crate) metal: bool,
    pub(crate) vulkan: bool,
    pub(crate) openmp: bool,
    pub(crate) pthreads: bool,
}

impl FeatureFlags {
    fn from_env() -> Self {
        Self {
            backend_dl: env_flag("CARGO_FEATURE_BACKEND_DL"),
            cuda: env_flag("CARGO_FEATURE_CUDA"),
            metal: env_flag("CARGO_FEATURE_METAL"),
            vulkan: env_flag("CARGO_FEATURE_VULKAN"),
            openmp: env_flag("CARGO_FEATURE_OPENMP"),
            pthreads: env_flag("CARGO_FEATURE_PTHREADS") || env_flag("CARGO_FEATURE_PTHREAD"),
        }
    }

    pub(crate) fn backend_tag(&self) -> String {
        let backend = if self.cuda {
            "cu"
        } else if self.metal {
            "mt"
        } else if self.vulkan {
            "vk"
        } else if self.openmp {
            "om"
        } else {
            "c"
        };

        if self.backend_dl {
            format!("dl-{backend}")
        } else {
            backend.to_owned()
        }
    }
}

pub(crate) struct BuildEnv {
    pub(crate) cuda_path: Option<PathBuf>,
    pub(crate) cuda_architectures: Option<String>,
    pub(crate) vulkan_sdk: Option<PathBuf>,
    pub(crate) glslc: Option<PathBuf>,
    pub(crate) cmake_out_dir: Option<PathBuf>,
    pub(crate) static_cxx_runtime_lib_dir: Option<PathBuf>,
}

impl BuildEnv {
    fn from_env() -> Self {
        Self {
            cuda_path: env::var_os("CUDA_PATH")
                .or_else(|| env::var_os("CUDA_HOME"))
                .map(PathBuf::from),
            cuda_architectures: env::var("SIPP_CUDA_ARCHITECTURES")
                .ok()
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty()),
            vulkan_sdk: env::var_os("VULKAN_SDK").map(PathBuf::from),
            glslc: env_path("SIPP_GLSLC"),
            cmake_out_dir: env::var("SIPP_SYS_CMAKE_OUT_DIR").ok().map(sanitize_path),
            static_cxx_runtime_lib_dir: env_path("SIPP_STATIC_CXX_RUNTIME_LIB_DIR"),
        }
    }
}

fn env_flag(name: &str) -> bool {
    env::var_os(name).is_some()
}

fn env_path(name: &str) -> Option<PathBuf> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
        .map(sanitize_path)
}
