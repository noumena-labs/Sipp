use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let llama_dir = manifest_dir.join("../../../third_party/llama.cpp");
    assert!(
        llama_dir.join("include/llama.h").exists(),
        "vendored llama.cpp directory"
    );

    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=include/cogent_shim.h");
    println!("cargo:rerun-if-changed=src/cogent_shim.cpp");
    println!("cargo:rerun-if-changed=src/bindings_fallback.rs");
    println!("cargo:rerun-if-env-changed=COGENTLM_SYS_GENERATE_BINDINGS");
    println!("cargo:rerun-if-env-changed=COGENTLM_SYS_SKIP_CMAKE");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    if env::var_os("CARGO_FEATURE_NATIVE").is_some()
        && env::var_os("COGENTLM_SYS_SKIP_CMAKE").is_none()
    {
        build_native(&manifest_dir, &llama_dir);
    }

    generate_or_copy_bindings(&manifest_dir, &llama_dir);
}

fn build_native(manifest_dir: &Path, llama_dir: &Path) {
    let mut config = cmake::Config::new(manifest_dir);
    config
        .profile("Release")
        .define("COGENTLM_LLAMA_CPP_DIR", llama_dir)
        .define("CMAKE_INSTALL_LIBDIR", "lib");

    define_bool_feature(&mut config, "CARGO_FEATURE_CUDA", "GGML_CUDA");
    define_bool_feature(&mut config, "CARGO_FEATURE_METAL", "GGML_METAL");
    define_bool_feature(&mut config, "CARGO_FEATURE_VULKAN", "GGML_VULKAN");
    define_bool_feature(&mut config, "CARGO_FEATURE_OPENMP", "GGML_OPENMP");

    let dst = config.build();
    let lib_dir = if dst.join("lib").exists() {
        dst.join("lib")
    } else {
        dst.join("lib64")
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    for lib in [
        "cogent_shim",
        "mtmd",
        "llama-common",
        "llama-common-base",
        "cpp-httplib",
        "llama",
        "ggml",
        "ggml-cpu",
        "ggml-base",
        "ggml-cuda",
        "ggml-metal",
        "ggml-vulkan",
        "ggml-blas",
    ] {
        if static_library_exists(&lib_dir, lib) {
            println!("cargo:rustc-link-lib=static={lib}");
        }
    }

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "windows" => {
            for lib in [
                "ws2_32", "bcrypt", "userenv", "advapi32", "ole32", "shell32", "uuid",
            ] {
                println!("cargo:rustc-link-lib={lib}");
            }
            if env::var_os("CARGO_FEATURE_CUDA").is_some() {
                link_cuda_libraries_windows();
            }
        }
        "macos" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            if env::var_os("CARGO_FEATURE_METAL").is_some() {
                for framework in [
                    "Foundation",
                    "Metal",
                    "MetalKit",
                    "QuartzCore",
                    "CoreGraphics",
                ] {
                    println!("cargo:rustc-link-lib=framework={framework}");
                }
            }
        }
        _ => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
            if env::var_os("CARGO_FEATURE_CUDA").is_some() {
                link_cuda_libraries_unix();
            }
        }
    }
}

fn link_cuda_libraries_windows() {
    if let Some(cuda_path) = cuda_path() {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_path.join("lib/x64").display()
        );
    }

    for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn link_cuda_libraries_unix() {
    if let Some(cuda_path) = cuda_path() {
        println!(
            "cargo:rustc-link-search=native={}",
            cuda_path.join("lib64").display()
        );
    }

    for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

fn cuda_path() -> Option<PathBuf> {
    env::var_os("CUDA_PATH")
        .or_else(|| env::var_os("CUDA_HOME"))
        .map(PathBuf::from)
}

fn define_bool_feature(config: &mut cmake::Config, feature_env: &str, cmake_name: &str) {
    config.define(
        cmake_name,
        if env::var_os(feature_env).is_some() {
            "ON"
        } else {
            "OFF"
        },
    );
}

fn static_library_exists(lib_dir: &Path, lib: &str) -> bool {
    let names = [
        format!("{lib}.lib"),
        format!("lib{lib}.a"),
        format!("lib{lib}.lib"),
        format!("{lib}.a"),
    ];
    names.iter().any(|name| lib_dir.join(name).exists())
}

fn generate_or_copy_bindings(manifest_dir: &Path, llama_dir: &Path) {
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("bindings.rs");
    let fallback = manifest_dir.join("src/bindings_fallback.rs");
    let generate = env::var("COGENTLM_SYS_GENERATE_BINDINGS")
        .map(|value| value != "0")
        .unwrap_or(true);

    if generate {
        let result = std::panic::catch_unwind(|| {
            bindgen::Builder::default()
                .header(manifest_dir.join("wrapper.h").display().to_string())
                .allowlist_function("llama_.*")
                .allowlist_function("cogent_.*")
                .allowlist_type("llama_.*")
                .allowlist_type("ggml_.*")
                .allowlist_type("cogent_.*")
                .allowlist_var("LLAMA_.*")
                .allowlist_var("GGML_.*")
                .clang_arg(format!("-I{}", llama_dir.join("include").display()))
                .clang_arg(format!("-I{}", llama_dir.join("ggml/include").display()))
                .clang_arg(format!("-I{}", manifest_dir.join("include").display()))
                .derive_default(true)
                .layout_tests(false)
                .generate()
        });

        if let Ok(Ok(bindings)) = result {
            bindings
                .write_to_file(&out_path)
                .expect("write generated bindings");
            return;
        }

        println!(
            "cargo:warning=bindgen failed or libclang is unavailable; using checked-in fallback bindings"
        );
    }

    fs::copy(&fallback, &out_path).expect("copy fallback bindings");
}
