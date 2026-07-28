//! Developer environment readiness checks.

use crate::cli::{Backend, DoctorArgs, DoctorTarget};
use crate::output;
use crate::targets::swift::{CODESIGN_PATH, DITTO_PATH, PLUTIL_PATH, XCRUN_PATH};
use crate::toolchain::{self, ToolStatus};
use crate::toolchains::rustup::APPLE_TARGETS;
use crate::utils::BuildContext;
use anyhow::Result;
use std::path::Path;
use std::process::Command;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "tests/doctor_tests.rs"]
mod doctor_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
/// Runs read-only developer environment checks.
pub fn run(ctx: &BuildContext, args: &DoctorArgs) -> Result<()> {
    output::phase("Developer environment doctor");
    output::path("Workspace", ctx.workspace_root());
    output::path("Build root", &ctx.build_root());
    output::detail("Target", doctor_target_label(&args.target));
    output::detail("Backend", args.backend.as_str());

    let mut hard_failures = 0;

    if includes_core(&args.target) {
        hard_failures += print_core_statuses();
    }

    let node_included = includes_node(&args.target);

    if node_included {
        print_node_statuses(ctx);
    }

    if includes_python(&args.target) {
        print_python_statuses(ctx);
    }

    if includes_wasm(&args.target) {
        print_wasm_statuses(ctx, !node_included);
    }

    if includes_swift(&args.target) {
        hard_failures += print_swift_statuses(ctx);
    }

    if includes_native_backend(&args.target) {
        print_backend_statuses(ctx, &args.backend);
    }

    if hard_failures > 0 {
        let target = doctor_target_label(&args.target);
        anyhow::bail!(
            "doctor found {hard_failures} missing prerequisite(s); fix them and run `cargo xtask doctor --target {target}` again"
        );
    }

    output::success("Doctor complete");
    Ok(())
}

fn print_core_statuses() -> usize {
    output::phase("Core prerequisites");
    let statuses = vec![
        required_command_status("Cargo", "cargo", "Install Rust from https://rustup.rs/"),
        required_command_status("Rustc", "rustc", "Install Rust from https://rustup.rs/"),
    ];

    let mut failures = 0;
    for status in statuses {
        if status.is_missing() {
            failures += 1;
        }
        status.print();
    }

    toolchain::docker_status().print();

    failures
}

fn print_node_statuses(ctx: &BuildContext) {
    output::phase("Node binding readiness");
    toolchain::bun_status(ctx).print();
    toolchain::node_workspace_status(ctx).print();
    output::detail(
        "Recovery",
        "Run `cargo xtask setup --profile bindings --yes`",
    );
}

fn print_python_statuses(ctx: &BuildContext) {
    output::phase("Python binding readiness");
    toolchain::uv_status(ctx).print();
    output::detail("Recovery", "Run `cargo xtask toolchain install uv`");
}

fn print_wasm_statuses(ctx: &BuildContext, include_js_workspace: bool) {
    output::phase("WASM/browser readiness");
    if include_js_workspace {
        toolchain::bun_status(ctx).print();
    }
    toolchain::cmake_status(ctx).print();
    toolchain::ninja_status(ctx).print();
    toolchain::emsdk_status(ctx).print();
    if include_js_workspace {
        toolchain::node_workspace_status(ctx).print();
    }
    output::detail(
        "Recovery",
        "Run `cargo xtask setup --profile browser --yes`",
    );
}

fn print_swift_statuses(ctx: &BuildContext) -> usize {
    output::phase("Swift binding readiness");
    let statuses = vec![
        swift_host_status(),
        xcode_tool_status(
            "Xcode",
            "xcodebuild",
            "Install Xcode and select it with xcode-select",
        ),
        xcode_tool_status("Swift", "swift", "Install Swift through Xcode"),
        xcode_tool_status("Lipo", "lipo", "Install Xcode command-line tools"),
        xcode_tool_status("Nm", "nm", "Install Xcode command-line tools"),
        xcode_tool_status("Otool", "otool", "Install Xcode command-line tools"),
        system_tool_status(
            "Ditto",
            Path::new(DITTO_PATH),
            "Install macOS command-line tools",
        ),
        system_tool_status(
            "Codesign",
            Path::new(CODESIGN_PATH),
            "Install Xcode command-line tools",
        ),
        system_tool_status(
            "Plutil",
            Path::new(PLUTIL_PATH),
            "Install macOS command-line tools",
        ),
        swift_rust_targets_status(),
        toolchain::cmake_status(ctx),
        toolchain::ninja_status(ctx),
    ];

    let mut failures = 0;
    for status in statuses {
        if status.is_missing() {
            failures += 1;
        }
        status.print();
    }

    if ctx.llama_cpp_dir().join("include/llama.h").is_file() {
        output::success("llama.cpp submodule is initialized");
    } else {
        output::warning(
            "llama.cpp submodule is missing; run `git submodule update --init --recursive`",
        );
        failures += 1;
    }

    failures
}

fn print_backend_statuses(ctx: &BuildContext, backend: &Backend) {
    output::phase("Backend readiness");
    match backend {
        Backend::Cpu => output::success("CPU backend is always available"),
        Backend::Cuda => toolchain::cuda_status(ctx).print(),
        Backend::Metal => metal_status().print(),
        Backend::Vulkan => toolchain::vulkan_status(ctx).print(),
        Backend::All => {
            output::success("CPU backend is always available");
            if cfg!(target_os = "macos") {
                metal_status().print();
            } else {
                toolchain::vulkan_status(ctx).print();
                toolchain::cuda_status(ctx).print();
            }
        }
    }
}

fn required_command_status(
    name: &'static str,
    command: &'static str,
    fix: &'static str,
) -> ToolStatus {
    if toolchain::has_command(command) {
        ToolStatus::Ready {
            name,
            detail: format!("{command} is available"),
            path: None,
        }
    } else {
        ToolStatus::Missing {
            name,
            detail: format!("{command} is not available on PATH"),
            fix,
        }
    }
}

fn xcode_tool_status(name: &'static str, command: &'static str, fix: &'static str) -> ToolStatus {
    let output = Command::new(XCRUN_PATH).args(["--find", command]).output();
    match output {
        Ok(output) if output.status.success() => ToolStatus::Ready {
            name,
            detail: format!("{command} is available through the selected Xcode"),
            path: Some(String::from_utf8_lossy(&output.stdout).trim().into()),
        },
        Ok(output) => ToolStatus::Missing {
            name,
            detail: format!(
                "xcrun could not find {command}: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            fix,
        },
        Err(error) => ToolStatus::Missing {
            name,
            detail: format!("xcrun is not available: {error}"),
            fix,
        },
    }
}

fn system_tool_status(name: &'static str, path: &Path, fix: &'static str) -> ToolStatus {
    if path.is_file() {
        ToolStatus::Ready {
            name,
            detail: format!("{} is available", path.display()),
            path: Some(path.to_path_buf()),
        }
    } else {
        ToolStatus::Missing {
            name,
            detail: format!("{} is missing", path.display()),
            fix,
        }
    }
}

fn metal_status() -> ToolStatus {
    if cfg!(target_os = "macos") {
        ToolStatus::Ready {
            name: "Metal",
            detail: "host OS supports Metal backend builds".to_owned(),
            path: None,
        }
    } else {
        ToolStatus::Warn {
            name: "Metal",
            detail: "Metal backend builds require macOS".to_owned(),
            fix: "Use CPU, Vulkan, or CUDA on this host",
        }
    }
}

fn swift_host_status() -> ToolStatus {
    if cfg!(target_os = "macos") {
        ToolStatus::Ready {
            name: "macOS",
            detail: "host OS supports Apple package builds".to_owned(),
            path: None,
        }
    } else {
        ToolStatus::Missing {
            name: "macOS",
            detail: "Swift XCFramework builds require macOS".to_owned(),
            fix: "Run the Swift build on a macOS host",
        }
    }
}

fn swift_rust_targets_status() -> ToolStatus {
    let output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output();
    let output = match output {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            return ToolStatus::Missing {
                name: "Rust Apple targets",
                detail: format!(
                    "rustup target discovery failed: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ),
                fix: "Install rustup and the Rust Apple targets",
            };
        }
        Err(error) => {
            return ToolStatus::Missing {
                name: "Rust Apple targets",
                detail: format!("rustup is not available: {error}"),
                fix: "Install rustup and the Rust Apple targets",
            };
        }
    };

    let installed = String::from_utf8_lossy(&output.stdout);
    let missing = missing_swift_rust_targets(&installed);
    if missing.is_empty() {
        ToolStatus::Ready {
            name: "Rust Apple targets",
            detail: APPLE_TARGETS.join(", "),
            path: None,
        }
    } else {
        ToolStatus::Missing {
            name: "Rust Apple targets",
            detail: format!("missing {}", missing.join(", ")),
            fix: "Run `cargo xtask toolchain install rust-apple`",
        }
    }
}

fn missing_swift_rust_targets(installed: &str) -> Vec<&'static str> {
    APPLE_TARGETS
        .iter()
        .copied()
        .filter(|target| !installed.lines().any(|line| line.trim() == *target))
        .collect()
}

fn includes_core(target: &DoctorTarget) -> bool {
    matches!(
        target,
        DoctorTarget::All | DoctorTarget::Core | DoctorTarget::Swift
    )
}

fn includes_node(target: &DoctorTarget) -> bool {
    matches!(target, DoctorTarget::All | DoctorTarget::Node)
}

fn includes_python(target: &DoctorTarget) -> bool {
    matches!(target, DoctorTarget::All | DoctorTarget::Python)
}

fn includes_wasm(target: &DoctorTarget) -> bool {
    matches!(target, DoctorTarget::All | DoctorTarget::Wasm)
}

fn includes_swift(target: &DoctorTarget) -> bool {
    matches!(target, DoctorTarget::All | DoctorTarget::Swift)
}

fn includes_native_backend(target: &DoctorTarget) -> bool {
    matches!(
        target,
        DoctorTarget::All | DoctorTarget::Node | DoctorTarget::Python
    )
}

fn doctor_target_label(target: &DoctorTarget) -> &'static str {
    match target {
        DoctorTarget::All => "all",
        DoctorTarget::Core => "core",
        DoctorTarget::Wasm => "wasm",
        DoctorTarget::Node => "node",
        DoctorTarget::Python => "python",
        DoctorTarget::Swift => "swift",
    }
}
