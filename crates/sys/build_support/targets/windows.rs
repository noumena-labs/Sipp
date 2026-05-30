use crate::build_support::context::BuildContext;
use cmake::Config;

pub(crate) fn apply_host_cmake_overrides(context: &BuildContext, config: &mut Config) {
    if context.features.backend_dl {
        config.define("CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS", "ON");
    }

    if context.env_vars.cmake_out_dir.is_none() {
        let target_prefix = if context.target_kind.is_emscripten() {
            if context.features.pthreads {
                "wm_p"
            } else {
                "wm_s"
            }
        } else {
            "nt"
        };

        config.out_dir(
            context
                .manifest_dir
                .join("../../.build/c")
                .join(target_prefix)
                .join(context.features.backend_tag()),
        );
    }

    config.generator("Ninja");

    if context.target.contains("msvc") {
        config.define("CMAKE_POLICY_DEFAULT_CMP0194", "OLD");
        config.cxxflag("/EHsc");
        config.cxxflag("/FIistream");
    }

    if context.features.vulkan {
        if let Some(vulkan_sdk) = &context.env_vars.vulkan_sdk {
            config.define("Vulkan_ROOT", vulkan_sdk.as_os_str());
        }
    }
}

pub(crate) fn apply_cuda_cmake_overrides(config: &mut Config) {
    config.define("CMAKE_CUDA_FLAGS", "-allow-unsupported-compiler");
}

pub(crate) fn link_system_libraries(context: &BuildContext) {
    for lib in [
        "ws2_32", "bcrypt", "userenv", "advapi32", "ole32", "shell32", "uuid",
    ] {
        println!("cargo:rustc-link-lib={lib}");
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
        let lib_dir = vulkan_sdk.join("Lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    println!("cargo:rustc-link-lib=vulkan-1");
}

fn link_cuda_libraries(context: &BuildContext) {
    if let Some(cuda_path) = &context.env_vars.cuda_path {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_path.join("lib/x64").display()
        );
    }
    for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
