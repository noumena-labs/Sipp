use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));

    // Maintainers: If you restructure the workspace, ensure this relative path
    // correctly points to the vendored llama.cpp directory from this crate's root.
    let llama_dir = manifest_dir.join("../../third_party/llama.cpp");

    assert!(
        llama_dir.join("include/llama.h").exists(),
        "vendored llama.cpp directory must exist at {llama_dir:?}"
    );

    // If you add new C/C++ source files or headers to our shim, append them here.
    println!("cargo:rerun-if-changed=CMakeLists.txt");
    println!("cargo:rerun-if-changed=src/wrapper.h");
    println!("cargo:rerun-if-changed=include/cogent_shim.h");
    println!("cargo:rerun-if-changed=src/cogent_shim.cpp");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    // Native C++ libraries via CMake.
    build_native(&manifest_dir, &llama_dir);

    // Generate the Rust FFI bindings.
    generate_bindings(&manifest_dir, &llama_dir);
}

fn build_native(manifest_dir: &Path, llama_dir: &Path) {
    let mut config = cmake::Config::new(manifest_dir);
    config
        .profile("Release")
        .define("COGENTLM_LLAMA_CPP_DIR", llama_dir)
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .define("BUILD_SHARED_LIBS", "OFF");

    if cfg!(windows) {
        // This drops the CMake cache into `CogentLM/target/cm` instead of the deep OUT_DIR (WIN file length isues)
        let short_build_dir = manifest_dir.join("../../target/cm");
        config.out_dir(short_build_dir);

        // Make this build reproducible even when invoked outside xtask.
        config.generator("Ninja");

        // CMake 4.1+ / 4.3 MSVC ASM policy compatibility for vendored llama.cpp/ggml.
        // This is mainly needed while upstream ggml still enables ASM in a way that
        // trips MSVC + newer CMake.
        config.define("CMAKE_POLICY_DEFAULT_CMP0194", "OLD");

        // cpp-httplib / MSVC STL uses exception-aware code paths. Without this,
        // MSVC emits C4530 and some builds may fail if warnings are promoted.
        config.cxxflag("/EHsc");

        // Existing workaround.
        config.cxxflag("/FIistream");

        // Help CMake's FindVulkan when building from Cargo/N-API.
        if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
            if let Some(vulkan_sdk) = env::var_os("VULKAN_SDK") {
                config.define("Vulkan_ROOT", PathBuf::from(vulkan_sdk));
            }
        }
    }

    // Map our Cargo.toml features to llama.cpp's CMake flags.
    // When adding new hardware backends (e.g., SYCL, ROCm), register them here.
    define_bool_feature(&mut config, "CARGO_FEATURE_CUDA", "GGML_CUDA");
    define_bool_feature(&mut config, "CARGO_FEATURE_METAL", "GGML_METAL");
    define_bool_feature(&mut config, "CARGO_FEATURE_VULKAN", "GGML_VULKAN");
    define_bool_feature(&mut config, "CARGO_FEATURE_OPENMP", "GGML_OPENMP");

    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("emscripten") {
        // WASM cannot use Visual Studio. Force Ninja.
        config.generator("Ninja");

        // Prevent cargo from injecting `/c emcc.bat` flags
        config.no_build_target(true);
        config.define("CMAKE_ASM_FLAGS", "");

        // Requires explicit pthread flags for CMake's FindThreads to succeed
        config.define("CMAKE_C_FLAGS", "-pthread");
        config.define("CMAKE_CXX_FLAGS", "-pthread");

        config.define("THREADS_PREFER_PTHREAD_FLAG", "ON");
        config.define("CMAKE_USE_PTHREADS_INIT", "1");
        config.define("CMAKE_THREAD_LIBS_INIT", "-pthread");

        // Suppress a warning where CMake gets confused by Emscripten's archiver
        config.define("CMAKE_CXX_COMPILER_WORKS", "TRUE");
        config.define("CMAKE_C_COMPILER_WORKS", "TRUE");
    }

    let dst = config.build();

    // Setup clangd for static analysis
    setup_clangd_autocomplete(manifest_dir, &dst, llama_dir);

    let lib_dir = if dst.join("lib").exists() {
        dst.join("lib")
    } else {
        dst.join("lib64")
    };

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    // MSVC quirk: CMake will frequently nest outputs in build-type directories.
    // Do not remove these; it will break the Windows CI pipeline.
    println!(
        "cargo:rustc-link-search=native={}",
        lib_dir.join("Release").display()
    );
    println!(
        "cargo:rustc-link-search=native={}",
        lib_dir.join("Debug").display()
    );

    // This is the core list of static libraries produced by our CMake build.
    // If upstream llama.cpp renames or splits out new libraries, update this array.
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

    // Link OS-specific system libraries required by the underlying C++ code.
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
            if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
                link_vulkan_libraries_windows();
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
            if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
                link_vulkan_libraries_unix();
            }
        }
        "emscripten" => {
            // WASM manages its own WebGPU/WebGL bindings via Emscripten.
            // Do NOT link native host GPU libraries here.
        }
        _ => {
            // Linux / Unix defaults
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-lib=dylib=m");
            println!("cargo:rustc-link-lib=dylib=dl");
            println!("cargo:rustc-link-lib=dylib=pthread");
            if env::var_os("CARGO_FEATURE_CUDA").is_some() {
                link_cuda_libraries_unix();
            }
            if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
                link_vulkan_libraries_unix();
            }
        }
    }
}

fn link_vulkan_libraries_windows() {
    if let Some(vulkan_sdk) = env::var_os("VULKAN_SDK") {
        let lib_dir = PathBuf::from(vulkan_sdk).join("Lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    // MSVC expects the versioned `.lib` file
    println!("cargo:rustc-link-lib=vulkan-1");
}

fn link_vulkan_libraries_unix() {
    if let Some(vulkan_sdk) = env::var_os("VULKAN_SDK") {
        let lib_dir = PathBuf::from(vulkan_sdk).join("lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
    }
    // Unix/Linux uses the standard unversioned name
    println!("cargo:rustc-link-lib=vulkan");
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
    // Check standard environment variables. If you encounter a bespoke HPC
    // environment where these aren't set, add the custom ENV keys here.
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
    // Check standard UNIX formats alongside MSVC specific outputs (.lib).
    let names = [
        format!("{lib}.lib"),
        format!("lib{lib}.a"),
        format!("lib{lib}.lib"),
        format!("{lib}.a"),
    ];

    names.iter().any(|name| {
        lib_dir.join(name).exists()
            || lib_dir.join("Release").join(name).exists()
            || lib_dir.join("Debug").join(name).exists()
    })
}

fn generate_bindings(manifest_dir: &Path, llama_dir: &Path) {
    let out_path = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("bindings.rs");

    // Compute the platform-specific cache path
    let target = env::var("TARGET").unwrap_or_default();
    let target_filename = format!("{}.rs", target.replace("-", "_"));
    let pregenerated_path = manifest_dir.join("src/bindings").join(&target_filename);

    // Flag to for force binding generation
    let force_generate = env::var("COGENT_GENERATE_BINDINGS").is_ok();

    // Fast Path: If cached bindings exist and we aren't forcing a rebuild, use them!
    if !force_generate && pregenerated_path.exists() {
        println!("cargo:warning=Using pre-generated bindings for {target} from source tree.");
        std::fs::copy(&pregenerated_path, &out_path)
            .expect("Failed to copy pre-generated bindings to OUT_DIR");
        return;
    }

    // Fallback / Maintenance Path: Dynamically generate with bindgen (Requires LLVM)
    println!("cargo:warning=Dynamically generating bindings via libclang for {target}...");

    // Note: We heavily filter what bindgen generates to keep compile times fast
    // and the resulting file clean. If you need a new struct or function exposed
    // in Rust, ensure it matches one of these allowlist regexes.
    let mut builder = bindgen::Builder::default()
        .header(manifest_dir.join("src/wrapper.h").display().to_string())
        .allowlist_function("llama_.*")
        .allowlist_function("cogent_.*")
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .allowlist_type("cogent_.*")
        .allowlist_type("cogent_.*")
        .allowlist_var("LLAMA_.*")
        .allowlist_var("GGML_.*")
        .clang_arg(format!("-I{}", llama_dir.join("include").display()))
        .clang_arg(format!("-I{}", llama_dir.join("ggml/include").display()))
        .clang_arg(format!("-I{}", manifest_dir.join("include").display()))
        // .clang_arg("-v") // Verbose Clang output
        // .clang_arg("-Wno-everything") // Clear noise
        // .clang_arg("-Wall") // Only show actual compilation warnings/errors
        .derive_default(true)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));

    if target.contains("emscripten") {
        if let Ok(emsdk) = env::var("EMSDK") {
            let emsdk_path = PathBuf::from(emsdk);

            // 1. Get the standard sysroot include path
            let sysroot_include = emsdk_path
                .join("upstream/emscripten/cache/sysroot/include")
                .display()
                .to_string()
                .replace("\\", "/");

            // 2. Ask Emscripten's bundled Clang exactly where its internal intrinsic headers live
            let clang_exe = if cfg!(windows) { "clang.exe" } else { "clang" };
            let clang_path = emsdk_path.join("upstream/bin").join(clang_exe);

            let resource_dir_output = std::process::Command::new(clang_path)
                .arg("-print-resource-dir")
                .output()
                .expect("Failed to ask Emscripten clang for resource dir");

            let resource_dir = String::from_utf8_lossy(&resource_dir_output.stdout)
                .trim()
                .replace("\\", "/");

            builder = builder
                .clang_arg("--target=wasm32-unknown-emscripten")
                .clang_arg("-D__EMSCRIPTEN__")
                .clang_arg(format!("-I{}", sysroot_include))
                .clang_arg(format!("-resource-dir={}", resource_dir))
                .clang_arg("-fvisibility=default");
        }
    }

    let bindings = builder
        .generate()
        .expect("bindgen failed to generate bindings! Ensure libclang is installed.");

    bindings
        .write_to_file(&out_path)
        .expect("write generated bindings");

    // 4. If this pass was run with the update flag, save the bindings back into Git cache
    if force_generate {
        println!("cargo:warning=Saving newly generated bindings to src/bindings/{target_filename}");
        std::fs::create_dir_all(pregenerated_path.parent().unwrap()).unwrap();
        std::fs::copy(&out_path, &pregenerated_path)
            .expect("Failed to save generated bindings to source tree");
    }
}

// Convience  method for creating and setting up paths for clangd language server
fn setup_clangd_autocomplete(manifest_dir: &Path, dst: &Path, llama_dir: &Path) {
    let comp_db = dst.join("build/compile_commands.json");

    if comp_db.exists() {
        // Unix / Ninja: Copy generated database for IDE to use
        let _ = std::fs::copy(comp_db, manifest_dir.join("compile_commands.json"));
    } else {
        // Windows MSVC: Dynamically generate compile_flags.txt as a fallback for IDE
        let flags_path = manifest_dir.join("compile_flags.txt");
        if let Ok(mut file) = std::fs::File::create(flags_path) {
            let _ = writeln!(file, "-xc++");
            let _ = writeln!(file, "-std=c++17");

            // Format paths dynamically so they always match the host machine
            let _ = writeln!(file, "-I{}", manifest_dir.join("include").display());
            let _ = writeln!(file, "-I{}", llama_dir.join("include").display());
            let _ = writeln!(file, "-I{}", llama_dir.join("ggml/include").display());
            let _ = writeln!(file, "-I{}", llama_dir.join("common").display());
            let _ = writeln!(file, "-I{}", llama_dir.join("tools/mtmd").display());
            let _ = writeln!(file, "-I{}", llama_dir.join("vendor").display());

            // Add any specific macros your IDE needs to know about
            let _ = writeln!(file, "-DGGML_USE_OPENMP=OFF");
        }
    }
}
