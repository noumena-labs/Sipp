fn main() {
    let mut build = cxx_build::bridge("src/bridge.rs");
    build.file("src/bridge_shim.cpp").include("src");

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
    println!("cargo:rerun-if-changed=src/bridge_shim.cpp");
    println!("cargo:rerun-if-changed=src/bridge_shim.h");
    println!("cargo:rerun-if-changed=src/bridge_shim_c_api.h");
    println!("cargo:rerun-if-changed=src/api/ffi_types.h");
}
