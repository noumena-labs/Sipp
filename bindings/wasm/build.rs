fn main() {
    println!("cargo:rerun-if-changed=native/emscripten/ce_host.js");
}
