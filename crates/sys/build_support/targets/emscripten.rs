use crate::build_support::context::BuildContext;
use std::process::Command;

pub(crate) fn link_system_libraries() {}

pub(crate) fn apply_bindgen_args(
    context: &BuildContext,
    mut builder: bindgen::Builder,
) -> bindgen::Builder {
    let Some(emsdk_path) = &context.env_vars.emsdk else {
        return builder;
    };

    let sysroot_include = emsdk_path
        .join("upstream/emscripten/cache/sysroot/include")
        .display()
        .to_string()
        .replace('\\', "/");

    let clang_exe = if context.host_is_windows {
        "clang.exe"
    } else {
        "clang"
    };
    let clang_path = emsdk_path.join("upstream/bin").join(clang_exe);
    let resource_dir_output = Command::new(clang_path)
        .arg("-print-resource-dir")
        .output()
        .expect("Failed to ask Emscripten clang for resource dir");

    let resource_dir = String::from_utf8_lossy(&resource_dir_output.stdout)
        .trim()
        .replace('\\', "/");

    builder = builder
        .clang_arg("--target=wasm32-unknown-emscripten")
        .clang_arg("-D__EMSCRIPTEN__")
        .clang_arg(format!("-I{}", sysroot_include))
        .clang_arg(format!("-resource-dir={}", resource_dir))
        .clang_arg("-fvisibility=default");

    builder
}
