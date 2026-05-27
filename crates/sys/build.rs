// crates/sys/build.rs
fn main() {
    let dst = cmake::Config::new(".")
        .define("BUILD_SHARED_LIBS", "OFF")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=mtmd");
    println!("cargo:rustc-link-lib=static=cogent_shim");
    println!("cargo:rustc-link-lib=static=llama");
}
