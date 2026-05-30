use crate::build_support::context::BuildContext;

pub(crate) fn link_system_libraries(context: &BuildContext) {
    println!("cargo:rustc-link-lib=dylib=stdc++");
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
