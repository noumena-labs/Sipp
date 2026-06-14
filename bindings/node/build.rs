fn main() {
    napi_build::setup();

    // `napi_build::setup()` configures the cdylib that Node dlopen()s at
    // runtime. The `--lib` unit-test binary (built by `cargo llvm-cov --lib`
    // for coverage) is a plain executable with no Node host, so the napi_*
    // symbols the `#[napi]` exports reference are undefined and the link fails
    // on Linux. They are legitimately provided by Node at load time, so tell
    // the linker to tolerate undefined symbols.
    //
    // `rustc-link-arg` is the only instruction that reaches the lib unit-test
    // link — `rustc-link-arg-tests` applies only to `tests/` integration
    // targets, which this crate has none of. It also applies to the cdylib,
    // but that is a no-op there: a cdylib already links with undefined symbols
    // allowed. Windows resolves these via Node delay-load, so it needs no flag.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" | "android" => {
            println!("cargo:rustc-link-arg=-Wl,--unresolved-symbols=ignore-all");
        }
        "macos" | "ios" => {
            println!("cargo:rustc-link-arg=-undefined");
            println!("cargo:rustc-link-arg=dynamic_lookup");
        }
        _ => {}
    }
}
