//! Environment application for backend-specific native builds.

use crate::cli::Backend;
use crate::output;
use crate::toolchains::cuda::{cuda_architectures, setup_cuda};
use crate::toolchains::vulkan::{setup_vulkan, VULKAN_VERSION};
use crate::utils::BuildContext;
use anyhow::Result;
use std::env as std_env;
use xshell::{Cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/toolchains/env_tests.rs"]
mod env_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Applies required toolchain environment variables to a command.
pub(crate) fn apply_toolchains<'a>(
    sh: &Shell,
    ctx: &BuildContext,
    mut command: Cmd<'a>,
    backend: Option<&Backend>,
) -> Result<Cmd<'a>> {
    let mut path_additions = Vec::new();

    let ninja_dir = ctx.ninja_toolchain_dir();
    if ctx.ninja_exe().exists() {
        path_additions.push(ninja_dir.display().to_string());
        command = command.env("CMAKE_GENERATOR", "Ninja");
    }

    let uv_dir = ctx
        .workspace_root()
        .join(".build")
        .join("toolchain")
        .join("uv");
    if uv_dir.exists() {
        path_additions.push(uv_dir.display().to_string());
    }

    let bun_dir = ctx.bun_toolchain_dir();
    if bun_dir.exists() {
        path_additions.push(bun_dir.display().to_string());
    }

    let cmake_bin_dir = ctx.cmake_bin_dir()?;
    if cmake_bin_dir.exists() {
        path_additions.push(cmake_bin_dir.display().to_string());
    }

    if let Some(deployment_target) = macos_deployment_target() {
        output::detail("macOS deployment target", deployment_target);
        command = command.env("MACOSX_DEPLOYMENT_TARGET", deployment_target);
    }

    match backend {
        Some(Backend::Vulkan) => {
            output::detail("Toolchain", "Vulkan");
            if use_system_vulkan() {
                output::detail("Vulkan SDK", "system");
            } else {
                let vulkan_dir = setup_vulkan(sh, ctx)?;
                let bin_path = if cfg!(windows) {
                    vulkan_dir.join("Bin")
                } else if cfg!(target_os = "macos") {
                    vulkan_dir.join("macOS").join("bin")
                } else {
                    vulkan_dir.join(VULKAN_VERSION).join("x86_64").join("bin")
                };
                path_additions.push(bin_path.display().to_string());

                let vulkan_sdk_path = if cfg!(windows) {
                    vulkan_dir.to_path_buf()
                } else if cfg!(target_os = "macos") {
                    vulkan_dir.join("macOS")
                } else {
                    vulkan_dir.join(VULKAN_VERSION).join("x86_64")
                };
                command = command.env("VULKAN_SDK", &vulkan_sdk_path);

                let current_cmake_prefix = std_env::var("CMAKE_PREFIX_PATH").unwrap_or_default();
                let separator = path_separator();
                let new_cmake_prefix = if current_cmake_prefix.is_empty() {
                    vulkan_sdk_path.display().to_string()
                } else {
                    format!(
                        "{}{separator}{}",
                        vulkan_sdk_path.display(),
                        current_cmake_prefix
                    )
                };
                command = command.env("CMAKE_PREFIX_PATH", new_cmake_prefix);
            }
        }
        Some(Backend::Cuda) => {
            output::detail("Toolchain", "CUDA");
            let cuda_path = setup_cuda()?;
            let bin_path = cuda_path.join("bin");
            path_additions.push(bin_path.display().to_string());

            let nvcc_exe = if cfg!(windows) {
                bin_path.join("nvcc.exe")
            } else {
                bin_path.join("nvcc")
            };
            command = command.env("CUDACXX", nvcc_exe.display().to_string());
            command = command.env("CUDA_TOOLKIT_ROOT_DIR", cuda_path.display().to_string());
            command = command.env("SIPP_CUDA_ARCHITECTURES", cuda_architectures(ctx));
        }
        Some(Backend::Metal) => output::detail("Toolchain", "Metal"),
        Some(Backend::Cpu) | Some(Backend::All) | None => {
            output::detail("Toolchain", "CPU/default")
        }
    }

    if !path_additions.is_empty() {
        let current_path = std_env::var("PATH").unwrap_or_default();
        let separator = path_separator();
        let new_path = format!(
            "{}{separator}{}",
            path_additions.join(separator),
            current_path
        );
        command = command.env("PATH", new_path);
    }

    Ok(command)
}

fn path_separator() -> &'static str {
    if cfg!(windows) {
        ";"
    } else {
        ":"
    }
}

fn use_system_vulkan() -> bool {
    let Some(value) = std_env::var_os("SIPP_USE_SYSTEM_VULKAN") else {
        return false;
    };
    matches!(
        value.to_string_lossy().trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn macos_deployment_target() -> Option<&'static str> {
    if !cfg!(target_os = "macos") {
        return None;
    }

    if cfg!(target_arch = "aarch64") {
        Some("11.0")
    } else {
        Some("10.15")
    }
}
