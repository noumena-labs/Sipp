//! Tests the `doctor` module in `xtask`.
//!
//! Covers target inclusion, labels, and command-status classification with
//! deterministic impossible command names instead of depending on host readiness.

use crate::cli::{Backend, DoctorTarget};
use crate::toolchain::ToolStatus;

use super::{
    doctor_target_label, includes_core, includes_native_backend, includes_node, includes_python,
    includes_wasm, metal_status, optional_command_status, required_command_status,
};

#[test]
fn target_inclusion_matrix_matches_doctor_scope() {
    assert!(includes_core(&DoctorTarget::All));
    assert!(includes_core(&DoctorTarget::Core));
    assert!(!includes_core(&DoctorTarget::Node));
    assert!(includes_node(&DoctorTarget::All));
    assert!(includes_node(&DoctorTarget::Node));
    assert!(includes_python(&DoctorTarget::Python));
    assert!(includes_wasm(&DoctorTarget::Wasm));
    assert!(includes_native_backend(&DoctorTarget::Node));
    assert!(includes_native_backend(&DoctorTarget::Python));
    assert!(!includes_native_backend(&DoctorTarget::Core));
}

#[test]
fn doctor_labels_are_stable() {
    assert_eq!(doctor_target_label(&DoctorTarget::All), "all");
    assert_eq!(doctor_target_label(&DoctorTarget::Core), "core");
    assert_eq!(doctor_target_label(&DoctorTarget::Wasm), "wasm");
    assert_eq!(doctor_target_label(&DoctorTarget::Node), "node");
    assert_eq!(doctor_target_label(&DoctorTarget::Python), "python");
}

#[test]
fn command_status_helpers_distinguish_required_and_optional_missing_tools() {
    let missing = "sipp-definitely-not-installed-command";
    let required = required_command_status("Required", missing, "fix required");
    let optional = optional_command_status("Optional", missing, "fix optional");

    assert!(matches!(
        required,
        ToolStatus::Missing {
            name: "Required",
            ..
        }
    ));
    assert!(required.is_missing());
    assert!(matches!(
        optional,
        ToolStatus::Warn {
            name: "Optional",
            ..
        }
    ));
    assert!(!optional.is_missing());
}

#[test]
fn metal_status_reflects_current_host_platform() {
    match metal_status() {
        ToolStatus::Ready { name, .. } => {
            assert!(cfg!(target_os = "macos"));
            assert_eq!(name, "Metal");
        }
        ToolStatus::Warn { name, .. } => {
            assert!(!cfg!(target_os = "macos"));
            assert_eq!(name, "Metal");
        }
        ToolStatus::Missing { .. } => panic!("metal is never a hard missing prerequisite"),
    }
}

#[test]
fn backend_labels_used_by_doctor_are_cli_labels() {
    assert_eq!(Backend::All.as_str(), "all");
    assert_eq!(Backend::Vulkan.as_str(), "vulkan");
}
