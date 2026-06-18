use crate::build_support::context::BuildContext;
use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn apply_cuda_cmake_overrides(config: &mut Config) {
    config.define("CMAKE_CUDA_FLAGS", cuda_cmake_flags());
}

pub(crate) fn link_system_libraries(context: &BuildContext) {
    if uses_static_stdcpp(context) {
        let dir = static_stdcpp_search_dir().unwrap_or_else(|| {
            panic!(
                "SIPP_STATIC_CXX_RUNTIME requires libstdc++.a; install libstdc++-static or set CXX to a compiler that can locate it"
            )
        });
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    println!("cargo:rustc-link-lib={}", stdcpp_link_kind(context));
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=pthread");

    if context.features.backend_dl {
        println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    }

    if !context.features.backend_dl && context.features.cuda {
        link_cuda_libraries(context);
    }
    if !context.features.backend_dl && context.features.vulkan {
        link_vulkan_libraries(context);
    }
}

fn link_vulkan_libraries(context: &BuildContext) {
    if let Some(vulkan_sdk) = &context.env_vars.vulkan_sdk {
        let lib_dir = vulkan_sdk.join("lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    println!("cargo:rustc-link-lib=vulkan");
}

fn link_cuda_libraries(context: &BuildContext) {
    if let Some(cuda_path) = &context.env_vars.cuda_path {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_path.join("lib64").display()
        );
    }
    for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

pub(super) fn cuda_cmake_flags() -> &'static str {
    "-Xcompiler=-fPIC"
}

pub(super) fn stdcpp_link_kind(context: &BuildContext) -> &'static str {
    if uses_static_stdcpp(context) {
        "static=stdc++"
    } else {
        "dylib=stdc++"
    }
}

fn uses_static_stdcpp(context: &BuildContext) -> bool {
    context.target.contains("linux") && context.env_vars.static_cxx_runtime
}

fn static_stdcpp_search_dir() -> Option<PathBuf> {
    cpp_compiler_candidates().find_map(|compiler| {
        let output = Command::new(&compiler)
            .arg("-print-file-name=libstdc++.a")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        static_stdcpp_search_dir_from_output(&output.stdout)
    })
}

fn cpp_compiler_candidates() -> impl Iterator<Item = String> {
    env::var("CXX")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .into_iter()
        .chain(["c++".to_owned(), "g++".to_owned()])
}

pub(super) fn static_stdcpp_search_dir_from_output(stdout: &[u8]) -> Option<PathBuf> {
    let path = String::from_utf8_lossy(stdout).trim().to_owned();
    if path.is_empty() || path == "libstdc++.a" {
        return None;
    }
    let path = Path::new(&path);
    if path.file_name()? != "libstdc++.a" || !path.exists() {
        return None;
    }
    path.parent().map(Path::to_path_buf)
}
