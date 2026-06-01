use crate::build_support::context::BuildContext;

pub(crate) fn compile_bridge(context: &BuildContext) {
    let mut build = cxx_build::bridge("src/bridge.rs");
    build
        .file("src/cogent_cxx.cpp")
        .include(context.manifest_dir.join("include"))
        .include(context.llama_dir.join("include"))
        .include(context.llama_dir.join("ggml/include"))
        .include(context.llama_dir.join("common"))
        .include(context.llama_dir.join("tools/mtmd"))
        .include(context.llama_dir.join("vendor"));

    if context.target_kind.is_emscripten() {
        build.flag("-std=c++17").flag("-fwasm-exceptions");
    } else if context.host_is_windows {
        build.flag("/std:c++17").flag("/EHsc");
    } else {
        build.flag_if_supported("-std=c++17");
    }

    build.compile("cogentlm_sys_cxxbridge");
}
