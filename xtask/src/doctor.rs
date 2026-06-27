//! Developer environment readiness checks.

use crate::cli::{Backend, DoctorArgs, DoctorTarget};
use crate::output;
use crate::toolchain::{self, ToolStatus};
use crate::utils::BuildContext;
use anyhow::Result;

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

    if includes_native_backend(&args.target) {
        print_backend_statuses(ctx, &args.backend);
    }

    if hard_failures > 0 {
        anyhow::bail!(
            "doctor found {hard_failures} missing core prerequisite(s); fix them and run `cargo xtask doctor` again"
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

fn includes_core(target: &DoctorTarget) -> bool {
    matches!(target, DoctorTarget::All | DoctorTarget::Core)
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
    }
}
