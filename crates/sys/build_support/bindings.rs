use crate::build_support::{context::BuildContext, targets};
use std::path::PathBuf;

pub(crate) fn generate(context: &BuildContext) {
    let out_path = context.out_dir.join("bindings.rs");
    let pregenerated_path = context.binding_cache_path();

    if !context.env_vars.force_generate_bindings && pregenerated_path.exists() {
        println!(
            "cargo:warning=Using pre-generated bindings for {} from source tree.",
            context.target
        );
        std::fs::copy(&pregenerated_path, &out_path)
            .expect("Failed to copy pre-generated bindings to OUT_DIR");
        return;
    }

    println!(
        "cargo:warning=Dynamically generating bindings via libclang for {}...",
        context.target
    );

    let mut builder = bindgen::Builder::default()
        .header(
            context
                .manifest_dir
                .join("src/wrapper.h")
                .display()
                .to_string(),
        )
        .allowlist_function("llama_.*")
        .allowlist_function("cogent_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("cogent_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .clang_arg(include_arg(context.llama_dir.join("include")))
        .clang_arg(include_arg(context.llama_dir.join("ggml/include")))
        .clang_arg(include_arg(context.manifest_dir.join("include")))
        .derive_default(true)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    builder = targets::apply_bindgen_target_args(context, builder);

    let bindings = builder
        .generate()
        .expect("bindgen failed to generate bindings! Ensure libclang is installed.");

    bindings
        .write_to_file(&out_path)
        .expect("write generated bindings");

    if context.env_vars.force_generate_bindings {
        println!(
            "cargo:warning=Saving newly generated bindings to src/bindings/{}",
            context.binding_cache_file_name()
        );
        if let Some(parent) = pregenerated_path.parent() {
            std::fs::create_dir_all(parent).expect("Failed to create bindings cache directory");
        }
        std::fs::copy(&out_path, &pregenerated_path)
            .expect("Failed to save generated bindings to source tree");
    }
}

fn include_arg(path: PathBuf) -> String {
    format!("-I{}", path.display())
}
