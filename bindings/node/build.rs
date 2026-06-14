fn main() {
    napi_build::setup();

    // `napi_build::setup()` only configures the cdylib that Node dlopen()s at
    // runtime. The `--lib` unit-test binary (built by `cargo llvm-cov --lib`
    // for coverage) is a plain executable with no Node host, so the napi_*
    // symbols referenced by the `#[napi]` exports are undefined. They are
    // legitimately provided by Node at load time, not missing, so tell the
    // linker to tolerate them — but only for test/bench binaries, leaving the
    // shipped cdylib untouched. Windows already resolves these via Node
    // delay-load, so it needs no flag.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" | "android" => {
            println!("cargo:rustc-link-arg-tests=-Wl,--unresolved-symbols=ignore-all");
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-arg-tests=-undefined");
            println!("cargo:rustc-link-arg-tests=dynamic_lookup");
        }
        _ => {}
    }
}
