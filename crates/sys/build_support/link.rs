use crate::build_support::{context::BuildContext, targets::TargetKind};
use std::path::{Path, PathBuf};

pub(crate) fn link_cmake_output(context: &BuildContext, dst: &Path) {
    let lib_dir = if dst.join("lib").exists() {
        dst.join("lib")
    } else {
        dst.join("lib64")
    };

    let search_dirs = library_search_dirs(dst, &lib_dir);
    for dir in &search_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }

    link_cmake_libraries(context, &search_dirs);
    crate::build_support::targets::link_system_libraries(context);
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

fn link_cmake_libraries(context: &BuildContext, search_dirs: &[PathBuf]) {
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

    if !context.features.backend_dl {
        if context.features.vulkan {
            core_libraries.push("ggml-vulkan");
        }
        if context.features.cuda {
            core_libraries.push("ggml-cuda");
        }
        if context.features.metal {
            core_libraries.push("ggml-metal");
        }
        core_libraries.push("ggml-blas");
    }

    let link_type = if context.features.backend_dl {
        "dylib"
    } else {
        "static"
    };

    for lib in core_libraries {
        if static_library_exists(search_dirs, lib)
            || dynamic_library_exists(context.target_kind, search_dirs, lib)
        {
            println!("cargo:rustc-link-lib={}={}", link_type, lib);
        }
    }
}

fn static_library_exists(search_dirs: &[PathBuf], lib: &str) -> bool {
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

fn dynamic_library_exists(target_kind: TargetKind, search_dirs: &[PathBuf], lib: &str) -> bool {
    let names = if target_kind == TargetKind::Windows {
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
