//! Tests the `toolchain` module in `xtask`.
//!
//! Covers status classification, managed cache path checks, CUDA environment
//! handling, and component labels with fake toolchain directories instead of
//! downloading or executing managed installers.

use crate::cli::ToolchainComponent;
use crate::test_support::{EnvGuard, TempDir};
use crate::toolchains::vulkan::VULKAN_VERSION;
use crate::utils::BuildContext;

use super::{
    component_label, cuda_status, emsdk_status, node_modules_roots, node_workspace_status,
    uv_status, vulkan_glslc_path, vulkan_status, ToolStatus,
};

#[test]
fn managed_statuses_report_missing_fake_toolchains() {
    let temp = TempDir::new("toolchain-missing");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    assert!(matches!(
        uv_status(&ctx),
        ToolStatus::Missing { name: "uv", .. }
    ));
    assert!(matches!(
        emsdk_status(&ctx),
        ToolStatus::Missing {
            name: "Emscripten",
            ..
        }
    ));
    assert!(matches!(
        vulkan_status(&ctx),
        ToolStatus::Warn {
            name: "Vulkan SDK",
            ..
        }
    ));
}

#[test]
fn emsdk_and_vulkan_statuses_detect_fake_cache_markers() {
    let temp = TempDir::new("toolchain-present");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    if cfg!(windows) {
        temp.write(".build/toolchain/emsdk/emsdk_env.bat", "");
        temp.write(".build/toolchain/vulkan/Bin/glslc.exe", "");
    } else if cfg!(target_os = "macos") {
        temp.write(".build/toolchain/emsdk/emsdk_env.sh", "");
        temp.write(".build/toolchain/vulkan/macOS/bin/glslc", "");
    } else {
        temp.write(".build/toolchain/emsdk/emsdk_env.sh", "");
        temp.write(
            format!(".build/toolchain/vulkan/{VULKAN_VERSION}/x86_64/bin/glslc"),
            "",
        );
    }

    assert!(matches!(
        emsdk_status(&ctx),
        ToolStatus::Ready {
            name: "Emscripten",
            ..
        }
    ));
    assert!(matches!(
        vulkan_status(&ctx),
        ToolStatus::Ready {
            name: "Vulkan SDK",
            ..
        }
    ));
    assert_eq!(
        vulkan_glslc_path(&ctx)
            .file_name()
            .and_then(|name| name.to_str()),
        Some(if cfg!(windows) { "glslc.exe" } else { "glslc" })
    );
}

#[test]
fn cuda_status_uses_environment_roots_without_running_nvcc() {
    {
        let _env = EnvGuard::new(&[("CUDA_PATH", None), ("CUDA_HOME", None)]);
        assert!(matches!(
            cuda_status(),
            ToolStatus::Warn { name: "CUDA", .. }
        ));
    }

    let temp = TempDir::new("toolchain-cuda");
    let nvcc = if cfg!(windows) {
        "bin/nvcc.exe"
    } else {
        "bin/nvcc"
    };
    temp.write(nvcc, "");
    let root = temp.path().display().to_string();
    let _env = EnvGuard::new(&[("CUDA_PATH", Some(&root)), ("CUDA_HOME", None)]);

    assert!(matches!(
        cuda_status(),
        ToolStatus::Ready { name: "CUDA", .. }
    ));
}

#[test]
fn node_workspace_status_checks_every_workspace_node_modules_root() {
    let temp = TempDir::new("toolchain-node-workspaces");
    temp.create_dir("demos/chat/node_modules");
    temp.create_dir("tools/playground/node_modules");
    temp.create_dir("lib/web/node_modules");
    temp.create_dir("lib/node/node_modules");
    temp.create_dir("bindings/node/node_modules");
    temp.create_dir("node_modules");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    assert!(matches!(
        node_workspace_status(&ctx),
        ToolStatus::Ready {
            name: "Node workspaces",
            ..
        }
    ));

    let roots = node_modules_roots(&ctx);
    assert!(roots.contains(&temp.join("node_modules")));
    assert!(roots.contains(&temp.join("bindings/node/node_modules")));
    assert!(roots.contains(&temp.join("demos/chat/node_modules")));
    assert!(roots.contains(&temp.join("tools/playground/node_modules")));
    assert!(roots.contains(&temp.join("lib/web/node_modules")));
    assert!(roots.contains(&temp.join("lib/node/node_modules")));
}

#[test]
fn component_labels_match_cli_values() {
    assert_eq!(component_label(&ToolchainComponent::All), "all");
    assert_eq!(component_label(&ToolchainComponent::Uv), "uv");
    assert_eq!(component_label(&ToolchainComponent::Ninja), "ninja");
    assert_eq!(component_label(&ToolchainComponent::Emsdk), "emsdk");
    assert_eq!(component_label(&ToolchainComponent::Vulkan), "vulkan");
}
