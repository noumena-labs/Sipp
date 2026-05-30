use std::env;
use std::io::Write;
use std::path::{Path, PathBuf};

fn sanitize_path(path_str: String) -> PathBuf {
    if path_str.starts_with(r"\\?\") {
        PathBuf::from(&path_str[4..])
    } else {
        PathBuf::from(path_str)
    }
}

fn main() {
    let manifest_dir_str = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let manifest_dir = sanitize_path(manifest_dir_str);

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
    println!("cargo:rerun-if-env-changed=COGENTLM_SYS_CMAKE_OUT_DIR");

    let target = env::var("TARGET").unwrap_or_default();

    // The browser build links C/C++ in bindings/wasm/CMakeLists.txt after
    // Cargo produces the Rust staticlib. Running this CMake phase here would
    // compile the same sources twice and can collide with native CMake caches.
    if !target.contains("emscripten") {
        build_native(&manifest_dir, &llama_dir);
    }

    // Generate the Rust FFI bindings.
    generate_bindings(&manifest_dir, &llama_dir);
}

fn build_native(manifest_dir: &Path, llama_dir: &Path) {
    let target = env::var("TARGET").unwrap_or_default();
    let backend_dl = env::var_os("CARGO_FEATURE_BACKEND_DL").is_some();
    let backend_tag = cmake_backend_tag();
    let cmake_out_dir = env::var("COGENTLM_SYS_CMAKE_OUT_DIR")
        .ok()
        .map(sanitize_path);

    let mut config = cmake::Config::new(manifest_dir);
    config
        .profile("Release")
        .define("COGENTLM_LLAMA_CPP_DIR", llama_dir)
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .define("BUILD_SHARED_LIBS", if backend_dl { "ON" } else { "OFF" })
        .define("GGML_BACKEND_DL", if backend_dl { "ON" } else { "OFF" })
        // Remove bloat from llama.cpp
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF");

    if backend_dl {
        config
            .define("GGML_NATIVE", "OFF")
            .define("GGML_CPU_ALL_VARIANTS", "ON");
    }

    let default_cmake_out_dir = workspace_build_dir(manifest_dir)
        .join("cmake")
        .join("sys")
        .join(path_component(&target, "host"))
        .join(backend_tag.as_str());
    let selected_cmake_out_dir = cmake_out_dir.clone().unwrap_or(default_cmake_out_dir);
    config.out_dir(selected_cmake_out_dir);

    if cfg!(windows) {
        // We only pass this on Windows to keep our cross-platform config perfectly clean.
        if backend_dl {
            config.define("CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS", "ON");
        }

        // Detect if Cargo is currently compiling the multi-threaded or single-threaded variant.
        // (Adjust the "CARGO_FEATURE_PTHREAD" string to match whatever feature name you use in xtask)
        let is_pthread =
            env::var("CARGO_FEATURE_PTHREADS").is_ok() || env::var("CARGO_FEATURE_PTHREAD").is_ok();

        // FIX: Isolate the single-threaded and multi-threaded CMake caches
        let target_prefix = if target.contains("emscripten") {
            if is_pthread {
                "wm_p"
            } else {
                "wm_s"
            }
        } else {
            "nt"
        };

        if cmake_out_dir.is_none() {
            let short_build_dir = manifest_dir
                .join("../../.build/c") // Shrink `.build/cmake/sys/target/` down to just `.b/c/`
                .join(target_prefix)
                .join(backend_tag.as_str());

            config.out_dir(short_build_dir);
        }
        config.generator("Ninja");

        if target.contains("msvc") {
            // CMake 4.1+ / 4.3 MSVC ASM policy compatibility for vendored llama.cpp/ggml.
            // This is mainly needed while upstream ggml still enables ASM in a way that
            // trips MSVC + newer CMake.
            config.define("CMAKE_POLICY_DEFAULT_CMP0194", "OLD");

            // cpp-httplib / MSVC STL uses exception-aware code paths. Without this,
            // MSVC emits C4530 and some builds may fail if warnings are promoted.
            config.cxxflag("/EHsc");

            // Existing workaround.
            config.cxxflag("/FIistream");
        }

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
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        // Targets: Pascal (61), Volta (70), Turing (75), Ampere (80, 86), Ada Lovelace (89), Hopper (90), Blackwell (100, 120)
        config.define("CMAKE_CUDA_ARCHITECTURES", "61;70;75;80;86;89;90;100;120");
        if cfg!(windows) {
            config.define("CMAKE_CUDA_FLAGS", "-allow-unsupported-compiler");
        }
    }

    define_bool_feature(&mut config, "CARGO_FEATURE_METAL", "GGML_METAL");
    define_bool_feature(&mut config, "CARGO_FEATURE_VULKAN", "GGML_VULKAN");
    define_bool_feature(&mut config, "CARGO_FEATURE_OPENMP", "GGML_OPENMP");

    let dst = config.build();

    // Setup clangd for static analysis
    setup_clangd_autocomplete(manifest_dir, &dst, llama_dir);

    let lib_dir = if dst.join("lib").exists() {
        dst.join("lib")
    } else {
        dst.join("lib64")
    };
    let search_dirs = library_search_dirs(&dst, &lib_dir);
    for dir in &search_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    link_cmake_libraries(&search_dirs, backend_dl);

    // Link OS-specific system libraries required by the underlying C++ code.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "windows" => {
            for lib in [
                "ws2_32", "bcrypt", "userenv", "advapi32", "ole32", "shell32", "uuid",
            ] {
                println!("cargo:rustc-link-lib={lib}");
            }
            if !backend_dl && env::var_os("CARGO_FEATURE_CUDA").is_some() {
                link_cuda_libraries_windows();
            }
            if !backend_dl && env::var_os("CARGO_FEATURE_VULKAN").is_some() {
                link_vulkan_libraries_windows();
            }
        }
        "macos" => {
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-lib=framework=Accelerate");
            if backend_dl {
                println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
            }
            if !backend_dl && env::var_os("CARGO_FEATURE_METAL").is_some() {
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
            if !backend_dl && env::var_os("CARGO_FEATURE_VULKAN").is_some() {
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
            if backend_dl {
                println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
            }
            if !backend_dl && env::var_os("CARGO_FEATURE_CUDA").is_some() {
                link_cuda_libraries_unix();
            }
            if !backend_dl && env::var_os("CARGO_FEATURE_VULKAN").is_some() {
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

fn cmake_backend_tag() -> String {
    let backend = if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        "cu"
    } else if env::var_os("CARGO_FEATURE_METAL").is_some() {
        "mt"
    } else if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        "vk"
    } else if env::var_os("CARGO_FEATURE_OPENMP").is_some() {
        "om"
    } else {
        "c"
    };

    if env::var_os("CARGO_FEATURE_BACKEND_DL").is_some() {
        format!("dl-{backend}")
    } else {
        backend.to_string()
    }
}

fn workspace_build_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("../../.build")
}

fn path_component(value: &str, fallback: &str) -> String {
    let source = if value.is_empty() { fallback } else { value };
    source
        .chars()
        .map(|ch| match ch {
            '/' | '\\' | ':' => '_',
            _ => ch,
        })
        .collect()
}

fn library_search_dirs(dst: &Path, lib_dir: &Path) -> Vec<PathBuf> {
    let candidates = [
        lib_dir.to_path_buf(),
        lib_dir.join("Release"),
        lib_dir.join("Debug"),
        dst.join("bin"),
        dst.join("bin").join("Release"),
        dst.join("bin").join("Debug"),
    ];
    let mut dirs = Vec::new();
    for dir in candidates {
        if dir.exists() && !dirs.iter().any(|existing| existing == &dir) {
            dirs.push(dir);
        }
    }
    dirs
}

fn link_cmake_libraries(search_dirs: &[PathBuf], backend_dl: bool) {
    let mut core_libraries = vec![
        "cogent_shim",
        "mtmd",
        "llama-common",
        "llama-common-base",
        "cpp-httplib",
        "llama",
        "ggml",
        "ggml-cpu",
        "ggml-base",
    ];

    if !backend_dl {
        if std::env::var_os("CARGO_FEATURE_VULKAN").is_some() {
            core_libraries.push("ggml-vulkan");
        }
        if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
            core_libraries.push("ggml-cuda");
        }
        if std::env::var_os("CARGO_FEATURE_METAL").is_some() {
            core_libraries.push("ggml-metal");
        }
        // If you ever enable BLAS, add it here too
        core_libraries.push("ggml-blas");
    }

    // Dynamic for CLI, Static for Node/WASM/Python
    let link_type = if backend_dl { "dylib" } else { "static" };

    for lib in core_libraries {
        if static_library_exists(search_dirs, lib) || dynamic_library_exists(search_dirs, lib) {
            println!("cargo:rustc-link-lib={}={}", link_type, lib);
        }
    }
}

fn static_library_exists(search_dirs: &[PathBuf], lib: &str) -> bool {
    // Check standard UNIX formats alongside MSVC specific outputs (.lib).
    let names = [
        format!("{lib}.lib"),
        format!("lib{lib}.a"),
        format!("lib{lib}.lib"),
        format!("{lib}.a"),
    ];

    search_dirs
        .iter()
        .any(|dir| names.iter().any(|name| dir.join(name).exists()))
}

fn dynamic_library_exists(search_dirs: &[PathBuf], lib: &str) -> bool {
    let names = if cfg!(windows) {
        vec![format!("{lib}.lib"), format!("lib{lib}.dll.a")]
    } else {
        vec![
            format!("lib{lib}.so"),
            format!("lib{lib}.dylib"),
            format!("lib{lib}.tbd"),
        ]
    };

    search_dirs
        .iter()
        .any(|dir| names.iter().any(|name| dir.join(name).exists()))
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

// Convenience method for creating and setting up paths for clangd language server
fn setup_clangd_autocomplete(manifest_dir: &Path, dst: &Path, llama_dir: &Path) {
    let comp_db = dst.join("build/compile_commands.json");
    let comp_flags = dst.join("build/compile_flags.txt");
    let ide_dir = workspace_build_dir(manifest_dir).join("ide").join("sys");
    let compile_commands_path = ide_dir.join("compile_commands.json");
    let compile_flags_path = ide_dir.join("compile_flags.txt");

    if comp_db.exists() {
        let _ = std::fs::create_dir_all(&ide_dir);
        let _ = std::fs::copy(&comp_db, &compile_commands_path);
        let _ = std::fs::remove_file(&compile_flags_path);
        return;
    }

    if !comp_flags.exists() {
        // Generate compile_flags.txt only as a fallback for IDE autocomplete.
        let _ = std::fs::create_dir_all(&ide_dir);
        if let Ok(mut file) = std::fs::File::create(&compile_flags_path) {
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
        let _ = std::fs::remove_file(&compile_commands_path);
    }
}
