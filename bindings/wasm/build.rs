fn main() {
    let mut build = cxx_build::bridge("src/bridge.rs");
    build
        .file("native/cxx_bridge/gguf_callbacks.cpp")
        .file("native/rust_api/browser_engine_api.cpp")
        .include("native/cxx_bridge")
        .include("native/rust_api")
        .include("native/js_api");

    let target = std::env::var("TARGET").unwrap_or_default();
    if target.contains("emscripten") {
        build.flag("-std=c++17").flag("-fwasm-exceptions");
    } else if target.contains("msvc") {
        build.flag("/std:c++17").flag("/EHsc");
    } else {
        build.flag_if_supported("-std=c++17");
    }

    build.compile("cogentlm_wasm_cxxbridge");

    println!("cargo:rerun-if-changed=src/bridge.rs");
    println!("cargo:rerun-if-changed=native/cxx_bridge/gguf_callbacks.cpp");
    println!("cargo:rerun-if-changed=native/cxx_bridge/gguf_callbacks.h");
    println!("cargo:rerun-if-changed=native/rust_api/browser_engine_api.cpp");
    println!("cargo:rerun-if-changed=native/rust_api/browser_engine_api.h");
    println!("cargo:rerun-if-changed=native/js_api/ffi_types.h");
}
