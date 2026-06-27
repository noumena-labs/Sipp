//! Toolchain status and bootstrap commands.

use crate::cli::{ToolchainCommands, ToolchainComponent, ToolchainSetupComponent};
use crate::output;
use crate::toolchains::{bun, cmake, cuda, emsdk, ninja, python, vulkan};
use crate::utils::BuildContext;
use anyhow::Result;
use std::env;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use xshell::Shell;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/toolchain_tests.rs"]
mod toolchain_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Readiness state for a developer toolchain.
#[derive(Clone, Debug)]
pub(crate) enum ToolStatus {
    Ready {
        name: &'static str,
        detail: String,
        path: Option<PathBuf>,
    },
    Missing {
        name: &'static str,
        detail: String,
        fix: &'static str,
    },
    Warn {
        name: &'static str,
        detail: String,
        fix: &'static str,
    },
}

impl ToolStatus {
    pub(crate) fn is_missing(&self) -> bool {
        matches!(self, ToolStatus::Missing { .. })
    }

    pub(crate) fn print(&self) {
        match self {
            ToolStatus::Ready { name, detail, path } => {
                output::success(format!("{name}: {detail}"));
                if let Some(path) = path {
                    output::path("Path", path);
                }
            }
            ToolStatus::Missing { name, detail, fix } => {
                output::warning(format!("{name}: {detail}"));
                output::detail("Fix", fix);
            }
            ToolStatus::Warn { name, detail, fix } => {
                output::warning(format!("{name}: {detail}"));
                output::detail("Optional fix", fix);
            }
        }
    }
}

/// Runs a toolchain management command.
pub fn run(sh: &Shell, ctx: &BuildContext, command: ToolchainCommands) -> Result<()> {
    match command {
        ToolchainCommands::Status => print_status(ctx),
        ToolchainCommands::Install { component } => install(sh, ctx, component),
        ToolchainCommands::Setup { component } => setup_component(ctx, component),
    }
}

pub(crate) fn print_status(ctx: &BuildContext) -> Result<()> {
    output::phase("Toolchain status");
    output::path("Toolchain cache", &ctx.toolchain_dir());

    for status in managed_statuses(ctx) {
        status.print();
    }
    for status in external_statuses(ctx) {
        status.print();
    }

    Ok(())
}

fn install(sh: &Shell, ctx: &BuildContext, component: ToolchainComponent) -> Result<()> {
    output::phase("Install toolchain");
    output::detail("Component", component_label(&component));

    match component {
        ToolchainComponent::All => {
            bun::setup_bun(sh, ctx)?;
            cmake::setup_cmake(sh, ctx)?;
            python::setup_uv(sh, ctx)?;
            ninja::setup_ninja(sh, ctx)?;
            emsdk::setup_emsdk(sh, ctx)?;
            vulkan::setup_vulkan(sh, ctx)?;
        }
        ToolchainComponent::Bun => {
            bun::setup_bun(sh, ctx)?;
        }
        ToolchainComponent::Cmake => {
            cmake::setup_cmake(sh, ctx)?;
        }
        ToolchainComponent::Uv => {
            python::setup_uv(sh, ctx)?;
        }
        ToolchainComponent::Ninja => {
            ninja::setup_ninja(sh, ctx)?;
        }
        ToolchainComponent::Emsdk => {
            emsdk::setup_emsdk(sh, ctx)?;
        }
        ToolchainComponent::Vulkan => {
            vulkan::setup_vulkan(sh, ctx)?;
        }
    }

    output::success("Toolchain install complete");
    Ok(())
}

pub(crate) fn managed_statuses(ctx: &BuildContext) -> Vec<ToolStatus> {
    vec![
        bun_status(ctx),
        cmake_status(ctx),
        uv_status(ctx),
        ninja_status(ctx),
        emsdk_status(ctx),
        vulkan_status(ctx),
    ]
}

pub(crate) fn external_statuses(ctx: &BuildContext) -> Vec<ToolStatus> {
    vec![
        command_status(
            "Cargo",
            "cargo",
            ["--version"],
            "Install Rust from https://rustup.rs/",
        ),
        command_status(
            "Rustc",
            "rustc",
            ["--version"],
            "Install Rust from https://rustup.rs/",
        ),
        cuda_status(ctx),
        node_workspace_status(ctx),
        docker_status(),
    ]
}

pub(crate) fn cmake_status(ctx: &BuildContext) -> ToolStatus {
    let cmake_exe = match ctx.cmake_exe() {
        Ok(cmake_exe) => cmake_exe,
        Err(error) => {
            return ToolStatus::Missing {
                name: "CMake",
                detail: error.to_string(),
                fix: "Run `cargo xtask toolchain install cmake`",
            };
        }
    };
    if cmake_exe.exists() {
        return command_path_status(
            "CMake",
            &cmake_exe,
            ["--version"],
            "Run `cargo xtask toolchain install cmake`",
        );
    }

    ToolStatus::Missing {
        name: "CMake",
        detail: "managed CMake executable is missing".to_owned(),
        fix: "Run `cargo xtask toolchain install cmake`",
    }
}

pub(crate) fn bun_status(ctx: &BuildContext) -> ToolStatus {
    let bun_exe = ctx.bun_exe();
    if bun_exe.exists() {
        let Some(version) = bun::bun_version(&bun_exe) else {
            return ToolStatus::Missing {
                name: "Bun",
                detail: format!("{} exists but did not run successfully", bun_exe.display()),
                fix: "Run `cargo xtask toolchain install bun`",
            };
        };
        if version == bun::BUN_VERSION {
            return ToolStatus::Ready {
                name: "Bun",
                detail: version,
                path: Some(bun_exe),
            };
        }
        return ToolStatus::Missing {
            name: "Bun",
            detail: format!(
                "managed Bun version is {version}; expected {}",
                bun::BUN_VERSION
            ),
            fix: "Run `cargo xtask toolchain install bun`",
        };
    }

    ToolStatus::Missing {
        name: "Bun",
        detail: "managed Bun executable is missing".to_owned(),
        fix: "Run `cargo xtask toolchain install bun`",
    }
}

pub(crate) fn uv_status(ctx: &BuildContext) -> ToolStatus {
    let uv_exe = ctx.uv_exe();
    if uv_exe.exists() {
        return command_path_status(
            "uv",
            &uv_exe,
            ["--version"],
            "Run `cargo xtask toolchain install uv`",
        );
    }

    ToolStatus::Missing {
        name: "uv",
        detail: "managed uv executable is missing".to_owned(),
        fix: "Run `cargo xtask toolchain install uv`",
    }
}

pub(crate) fn ninja_status(ctx: &BuildContext) -> ToolStatus {
    let ninja_exe = ctx.ninja_exe();
    if ninja_exe.exists() {
        return command_path_status(
            "Ninja",
            &ninja_exe,
            ["--version"],
            "Run `cargo xtask toolchain install ninja`",
        );
    }

    ToolStatus::Missing {
        name: "Ninja",
        detail: "managed Ninja executable is missing".to_owned(),
        fix: "Run `cargo xtask toolchain install ninja`",
    }
}

pub(crate) fn emsdk_status(ctx: &BuildContext) -> ToolStatus {
    let emsdk_dir = ctx.emsdk_dir();
    let marker = if cfg!(windows) {
        emsdk_dir.join("emsdk_env.bat")
    } else {
        emsdk_dir.join("emsdk_env.sh")
    };

    if marker.exists() {
        ToolStatus::Ready {
            name: "Emscripten",
            detail: "emsdk cache is present".to_owned(),
            path: Some(emsdk_dir),
        }
    } else {
        ToolStatus::Missing {
            name: "Emscripten",
            detail: "emsdk cache is missing".to_owned(),
            fix: "Run `cargo xtask toolchain install emsdk`",
        }
    }
}

pub(crate) fn vulkan_status(ctx: &BuildContext) -> ToolStatus {
    let vulkan_dir = ctx.vulkan_dir();
    let glslc = vulkan_glslc_path(ctx);

    if glslc.exists() {
        ToolStatus::Ready {
            name: "Vulkan SDK",
            detail: "managed Vulkan SDK is present".to_owned(),
            path: Some(vulkan_dir),
        }
    } else {
        ToolStatus::Warn {
            name: "Vulkan SDK",
            detail: "managed Vulkan SDK is missing".to_owned(),
            fix: "Run `cargo xtask toolchain install vulkan`",
        }
    }
}

pub(crate) fn cuda_status(ctx: &BuildContext) -> ToolStatus {
    let cuda_path = env::var_os("CUDA_PATH").or_else(|| env::var_os("CUDA_HOME"));
    let Some(path) = cuda_path else {
        return ToolStatus::Warn {
            name: "CUDA",
            detail: "CUDA_PATH/CUDA_HOME is not set".to_owned(),
            fix: "Install NVIDIA CUDA Toolkit and restart the terminal",
        };
    };

    let root = PathBuf::from(path);
    let nvcc = root
        .join("bin")
        .join(if cfg!(windows) { "nvcc.exe" } else { "nvcc" });
    if !nvcc.exists() {
        return ToolStatus::Warn {
            name: "CUDA",
            detail: format!("nvcc was not found under {}", root.display()),
            fix: "Install NVIDIA CUDA Toolkit and set CUDA_PATH/CUDA_HOME",
        };
    }

    let (arches, source) = cuda::cuda_architectures_with_source(ctx);
    let detail = match source.config_path() {
        Some(cfg_path) => format!(
            "Toolkit at {}, arch list [{}: {}] ({arches})",
            root.display(),
            source.label(),
            cfg_path.display(),
        ),
        None => format!(
            "Toolkit at {}, arch list [{}] ({arches})",
            root.display(),
            source.label(),
        ),
    };
    ToolStatus::Ready {
        name: "CUDA",
        detail,
        path: Some(root),
    }
}

pub(crate) fn node_workspace_status(ctx: &BuildContext) -> ToolStatus {
    let mut missing = Vec::new();
    for path in node_modules_roots(ctx) {
        if !path.exists() {
            missing.push(path);
        }
    }

    if missing.is_empty() {
        ToolStatus::Ready {
            name: "Node workspaces",
            detail: "workspace node_modules directories are present".to_owned(),
            path: None,
        }
    } else {
        let detail = missing
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        ToolStatus::Warn {
            name: "Node workspaces",
            detail: format!("missing dependency installs: {detail}"),
            fix: "Run `cargo xtask setup --profile full --yes`",
        }
    }
}

pub(crate) fn docker_status() -> ToolStatus {
    if has_command("docker") {
        ToolStatus::Ready {
            name: "Docker",
            detail: "Docker is available".to_owned(),
            path: None,
        }
    } else {
        ToolStatus::Warn {
            name: "Docker",
            detail: "Docker is not available on PATH".to_owned(),
            fix: "Install Docker from https://docker.com/",
        }
    }
}

pub(crate) fn has_command(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn command_status<const N: usize>(
    name: &'static str,
    command: &str,
    args: [&str; N],
    fix: &'static str,
) -> ToolStatus {
    match command_output(command, args) {
        Some(version) => ToolStatus::Ready {
            name,
            detail: version,
            path: None,
        },
        None => ToolStatus::Missing {
            name,
            detail: format!("{command} is not available on PATH"),
            fix,
        },
    }
}

fn command_path_status<const N: usize>(
    name: &'static str,
    path: &Path,
    args: [&str; N],
    fix: &'static str,
) -> ToolStatus {
    match command_path_output(path, args) {
        Some(version) => ToolStatus::Ready {
            name,
            detail: version,
            path: Some(path.to_path_buf()),
        },
        None => ToolStatus::Missing {
            name,
            detail: format!("{} exists but did not run successfully", path.display()),
            fix,
        },
    }
}

fn command_output<const N: usize>(command: &str, args: [&str; N]) -> Option<String> {
    command_output_inner(command, args)
}

fn command_path_output<const N: usize>(path: &Path, args: [&str; N]) -> Option<String> {
    command_output_inner(path.as_os_str(), args)
}

fn command_output_inner<const N: usize>(
    command: impl AsRef<OsStr>,
    args: [&str; N],
) -> Option<String> {
    let output = Command::new(command).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if !stdout.is_empty() {
        return Some(stdout);
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if stderr.is_empty() {
        Some("available".to_owned())
    } else {
        Some(stderr)
    }
}

fn node_modules_roots(ctx: &BuildContext) -> Vec<PathBuf> {
    let mut roots = vec![
        ctx.workspace_root().join("node_modules"),
        ctx.bindings_node_dir().join("node_modules"),
    ];

    if let Ok(dirs) = ctx.demo_dirs() {
        roots.extend(dirs.into_iter().map(|dir| dir.join("node_modules")));
    }
    if let Ok(dirs) = ctx.tool_dirs() {
        roots.extend(dirs.into_iter().map(|dir| dir.join("node_modules")));
    }
    roots.extend(
        ctx.js_package_dirs()
            .into_iter()
            .map(|dir| dir.join("node_modules")),
    );

    roots
}

fn vulkan_glslc_path(ctx: &BuildContext) -> PathBuf {
    let vulkan_dir = ctx.vulkan_dir();
    if cfg!(windows) {
        vulkan_dir.join("Bin").join("glslc.exe")
    } else if cfg!(target_os = "macos") {
        vulkan_dir.join("macOS").join("bin").join("glslc")
    } else {
        vulkan_dir
            .join(vulkan::VULKAN_VERSION)
            .join("x86_64")
            .join("bin")
            .join("glslc")
    }
}

fn setup_component(ctx: &BuildContext, component: ToolchainSetupComponent) -> Result<()> {
    match component {
        ToolchainSetupComponent::Cuda => cuda::setup_cuda_architectures(ctx),
    }
}

fn component_label(component: &ToolchainComponent) -> &'static str {
    match component {
        ToolchainComponent::All => "all",
        ToolchainComponent::Bun => "bun",
        ToolchainComponent::Cmake => "cmake",
        ToolchainComponent::Uv => "uv",
        ToolchainComponent::Ninja => "ninja",
        ToolchainComponent::Emsdk => "emsdk",
        ToolchainComponent::Vulkan => "vulkan",
    }
}
