use crate::build_support::context::BuildContext;
use cmake::Config;
use std::path::Path;

pub(crate) fn apply_cuda_cmake_overrides(config: &mut Config) {
    config.define("CMAKE_CUDA_FLAGS", cuda_cmake_flags());
}

pub(crate) fn link_system_libraries(context: &BuildContext) {
    if uses_static_stdcpp(context) {
        let dir = static_stdcpp_lib_dir(context);
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
    context.target.contains("linux") && context.env_vars.static_cxx_runtime_lib_dir.is_some()
}

pub(super) fn static_stdcpp_lib_dir(context: &BuildContext) -> &Path {
    let Some(dir) = context.env_vars.static_cxx_runtime_lib_dir.as_deref() else {
        panic!("SIPP_STATIC_CXX_RUNTIME_LIB_DIR is required for static libstdc++ linking");
    };
    let archive = dir.join("libstdc++.a");
    if !archive.exists() {
        panic!("SIPP_STATIC_CXX_RUNTIME_LIB_DIR does not contain libstdc++.a");
    }
    dir
}
