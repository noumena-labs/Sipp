use crate::build_support::{context::BuildContext, targets};
use cmake::Config;
use std::path::PathBuf;

pub(crate) fn build_native(context: &BuildContext) -> PathBuf {
    let mut config = Config::new(&context.manifest_dir);
    config
        .profile("Release")
        .define("SIPP_LLAMA_CPP_DIR", context.llama_dir.as_os_str())
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .define("BUILD_SHARED_LIBS", cmake_bool(context.features.backend_dl))
        .define("GGML_BACKEND_DL", cmake_bool(context.features.backend_dl))
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF");

    if context.features.backend_dl {
        config
            .define("GGML_NATIVE", "OFF")
            .define("GGML_CPU_ALL_VARIANTS", "ON");
    }

    config.out_dir(
        context
            .env_vars
            .cmake_out_dir
            .clone()
            .unwrap_or_else(|| context.default_cmake_out_dir()),
    );

    targets::apply_host_cmake_overrides(context, &mut config);

    define_bool_feature(&mut config, "GGML_CUDA", context.features.cuda);
    if context.features.cuda {
        // When unset, vendored llama.cpp selects CUDA-version-aware defaults.
        if let Some(cuda_architectures) = &context.env_vars.cuda_architectures {
            config.define("CMAKE_CUDA_ARCHITECTURES", cuda_architectures);
        }
        targets::apply_cuda_cmake_overrides(context, &mut config);
    }

    define_bool_feature(&mut config, "GGML_METAL", context.features.metal);
    define_bool_feature(&mut config, "GGML_VULKAN", context.features.vulkan);
    define_bool_feature(&mut config, "GGML_OPENMP", context.features.openmp);

    config.build()
}

fn define_bool_feature(config: &mut Config, cmake_name: &str, enabled: bool) {
    config.define(cmake_name, cmake_bool(enabled));
}

fn cmake_bool(enabled: bool) -> &'static str {
    if enabled {
        "ON"
    } else {
        "OFF"
    }
}
