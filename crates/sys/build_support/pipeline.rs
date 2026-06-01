use crate::build_support::{cmake, context::BuildContext, cxx, ide, link};

pub(crate) fn run() {
    let context = BuildContext::new();
    context.validate_llama_dir();
    context.emit_rerun_triggers();

    if !context.target_kind.is_emscripten() {
        let cmake_out_dir = cmake::build_native(&context);
        ide::setup_clangd_autocomplete(&context, &cmake_out_dir);
        cxx::compile_bridge(&context);
        link::link_cmake_output(&context, &cmake_out_dir);
    } else {
        cxx::compile_bridge(&context);
    }
}
