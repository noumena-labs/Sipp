//! Tests the `targets::cli` module in `xtask`.
//!
//! Covers backend expansion, runtime artifact classification, recursive
//! candidate collection, and copy filtering with fake CMake outputs instead of
//! compiling the CLI.

use crate::cli::Backend;
use crate::test_support::TempDir;

use super::{
    backend_label, backends_to_build, cargo_features, cli_binary_file_name,
    collect_runtime_candidates, copy_runtime_artifacts, is_backend_plugin_file,
    is_base_runtime_file, is_metal_sidecar_file, plugin_matches_backend, runtime_file_kind,
    RuntimeFileKind,
};

#[test]
fn backend_expansion_uses_host_supported_variants() {
    assert_eq!(backends_to_build(None), vec![Backend::Cpu]);
    assert_eq!(backends_to_build(Some(&Backend::Cuda)), vec![Backend::Cuda]);
    if cfg!(target_os = "macos") {
        assert_eq!(
            backends_to_build(Some(&Backend::All)),
            vec![Backend::Cpu, Backend::Metal]
        );
    } else {
        assert_eq!(
            backends_to_build(Some(&Backend::All)),
            vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda]
        );
    }
}

#[test]
fn labels_features_and_binary_names_are_stable() {
    assert_eq!(backend_label(None), "cpu (default)");
    assert_eq!(backend_label(Some(&Backend::Vulkan)), "vulkan");
    assert_eq!(cargo_features(&Backend::Cpu), "backend-dl");
    assert_eq!(cargo_features(&Backend::Cuda), "backend-dl,cuda");
    assert_eq!(
        cli_binary_file_name(),
        if cfg!(windows) {
            "sipp.exe"
        } else {
            "sipp"
        }
    );
}

#[test]
fn runtime_file_kind_classifies_base_plugins_and_metal_sidecars() {
    if cfg!(windows) {
        assert!(is_base_runtime_file("llama.dll"));
        assert!(is_backend_plugin_file("ggml-cpu.dll"));
        assert!(!is_backend_plugin_file("ggml-base.dll"));
        assert!(matches!(
            runtime_file_kind("llama.dll"),
            Some(RuntimeFileKind::Base)
        ));
    } else {
        assert!(is_base_runtime_file("libllama.so"));
        assert!(is_backend_plugin_file("libggml-cpu.so"));
        assert!(!is_backend_plugin_file("libggml-base.so"));
        assert!(matches!(
            runtime_file_kind("libllama.so"),
            Some(RuntimeFileKind::Base)
        ));
    }

    assert!(is_metal_sidecar_file("default.metallib"));
    assert!(matches!(
        runtime_file_kind("default.metallib"),
        Some(RuntimeFileKind::MetalSidecar)
    ));
    assert!(matches!(
        runtime_file_kind(if cfg!(windows) {
            "ggml-vulkan.dll"
        } else {
            "libggml-vulkan.so"
        }),
        Some(RuntimeFileKind::Plugin)
    ));
    assert!(runtime_file_kind("README.txt").is_none());
}

#[test]
fn plugin_matching_filters_backend_specific_runtime_files() {
    let cpu = if cfg!(windows) {
        "ggml-cpu.dll"
    } else {
        "libggml-cpu.so"
    };
    let cuda = if cfg!(windows) {
        "ggml-cuda.dll"
    } else {
        "libggml-cuda.so"
    };
    let metal = if cfg!(windows) {
        "ggml-metal.dll"
    } else {
        "libggml-metal.so"
    };
    let vulkan = if cfg!(windows) {
        "ggml-vulkan.dll"
    } else {
        "libggml-vulkan.so"
    };

    assert!(plugin_matches_backend(cpu, &Backend::Cpu));
    assert!(plugin_matches_backend(cuda, &Backend::Cuda));
    assert!(plugin_matches_backend(metal, &Backend::Metal));
    assert!(plugin_matches_backend(vulkan, &Backend::Vulkan));
    assert!(!plugin_matches_backend(vulkan, &Backend::All));
}

#[test]
fn runtime_candidate_collection_walks_known_output_roots() {
    let temp = TempDir::new("target-cli-candidates");
    temp.write("cmake/bin/b.txt", "");
    temp.write("cmake/lib/a.txt", "");
    temp.write("cmake/lib64/c.txt", "");
    temp.write("cmake/build/d.txt", "");
    temp.write("cmake/ignored/e.txt", "");

    let files = collect_runtime_candidates(&temp.join("cmake")).unwrap();

    assert_eq!(
        files,
        vec![
            temp.join("cmake/bin/b.txt"),
            temp.join("cmake/build/d.txt"),
            temp.join("cmake/lib/a.txt"),
            temp.join("cmake/lib64/c.txt")
        ]
    );
}

#[test]
fn runtime_copy_filters_base_plugins_and_sidecars() {
    let temp = TempDir::new("target-cli-copy");
    let sh = xshell::Shell::new().unwrap();
    let cmake = temp.create_dir("cmake");
    let dist = temp.create_dir("dist");
    let base = if cfg!(windows) {
        "llama.dll"
    } else {
        "libllama.so"
    };
    let cpu_plugin = if cfg!(windows) {
        "ggml-cpu.dll"
    } else {
        "libggml-cpu.so"
    };
    let vulkan_plugin = if cfg!(windows) {
        "ggml-vulkan.dll"
    } else {
        "libggml-vulkan.so"
    };
    temp.write(format!("cmake/bin/{base}"), "base");
    temp.write(format!("cmake/bin/{cpu_plugin}"), "cpu");
    temp.write(format!("cmake/lib/{vulkan_plugin}"), "vulkan");
    temp.write("cmake/build/default.metallib", "metal");

    let summary = copy_runtime_artifacts(&sh, &cmake, &dist, true, None).unwrap();

    assert_eq!(summary.base_files, 1);
    assert_eq!(summary.plugin_files, 2);
    assert!(summary.total_files >= 3);
    assert!(dist.join(base).exists());
    assert!(dist.join(cpu_plugin).exists());
    assert!(dist.join(vulkan_plugin).exists());

    let temp = TempDir::new("target-cli-copy-vulkan");
    let cmake = temp.create_dir("cmake");
    let dist = temp.create_dir("dist");
    temp.write(format!("cmake/bin/{base}"), "base");
    temp.write(format!("cmake/bin/{cpu_plugin}"), "cpu");
    temp.write(format!("cmake/bin/{vulkan_plugin}"), "vulkan");

    let summary =
        copy_runtime_artifacts(&sh, &cmake, &dist, false, Some(&Backend::Vulkan)).unwrap();

    assert_eq!(summary.base_files, 0);
    assert_eq!(summary.plugin_files, 1);
    assert!(!dist.join(base).exists());
    assert!(!dist.join(cpu_plugin).exists());
    assert!(dist.join(vulkan_plugin).exists());
}
