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
        // ggml defaults GGML_NATIVE=ON, which bakes in `-march=native` (the
        // build host's full ISA, including AVX-512). Binaries built that way
        // SIGILL when run on a narrower CPU — across heterogeneous CI runners,
        // and on end-user machines for distributed core/CLI/gateway artifacts.
        // Force it OFF so ggml targets a portable baseline: on x86 that is
        // SSE4.2/AVX/AVX2/BMI2/FMA/F16C (no AVX-512); other arches fall back to
        // their mandatory baseline. The DL build layers runtime variant
        // dispatch on top (below).
        .define("GGML_NATIVE", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF");

    if context.features.backend_dl {
        // GGML_CPU_ALL_VARIANTS requires GGML_BACKEND_DL: it builds every CPU
        // variant and selects one at load time.
        config.define("GGML_CPU_ALL_VARIANTS", "ON");
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
