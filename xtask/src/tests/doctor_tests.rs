//! Tests the `doctor` module in `xtask`.
//!
//! Covers target inclusion, labels, and command-status classification with
//! deterministic impossible command names instead of depending on host readiness.

use crate::cli::{Backend, DoctorTarget};
use crate::toolchain::ToolStatus;

use super::{
    doctor_target_label, includes_core, includes_native_backend, includes_node, includes_python,
    includes_swift, includes_wasm, metal_status, missing_swift_rust_targets,
    required_command_status, swift_host_status,
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
    assert!(includes_core(&DoctorTarget::Swift));
    assert!(includes_swift(&DoctorTarget::Swift));
    assert!(includes_swift(&DoctorTarget::All));
    assert!(!includes_native_backend(&DoctorTarget::Swift));
}

#[test]
fn doctor_labels_are_stable() {
    assert_eq!(doctor_target_label(&DoctorTarget::All), "all");
    assert_eq!(doctor_target_label(&DoctorTarget::Core), "core");
    assert_eq!(doctor_target_label(&DoctorTarget::Wasm), "wasm");
    assert_eq!(doctor_target_label(&DoctorTarget::Node), "node");
    assert_eq!(doctor_target_label(&DoctorTarget::Python), "python");
    assert_eq!(doctor_target_label(&DoctorTarget::Swift), "swift");
}

#[test]
fn required_command_status_reports_missing_tools() {
    let missing = "sipp-definitely-not-installed-command";
    let required = required_command_status("Required", missing, "fix required");

    assert!(matches!(
        required,
        ToolStatus::Missing {
            name: "Required",
            ..
        }
    ));
    assert!(required.is_missing());
}

#[test]
fn metal_status_reflects_current_host_platform() {
    #[cfg(target_os = "macos")]
    assert!(matches!(
        metal_status(),
        ToolStatus::Ready { name: "Metal", .. }
    ));

    #[cfg(not(target_os = "macos"))]
    assert!(matches!(
        metal_status(),
        ToolStatus::Warn { name: "Metal", .. }
    ));
}

#[test]
fn swift_host_status_requires_macos() {
    assert_eq!(swift_host_status().is_missing(), !cfg!(target_os = "macos"));
}

#[test]
fn swift_doctor_requires_all_apple_rust_targets() {
    let installed = "aarch64-apple-darwin\nx86_64-apple-darwin\naarch64-apple-ios\naarch64-apple-ios-sim\nx86_64-apple-ios\n";
    assert!(missing_swift_rust_targets(installed).is_empty());
    assert_eq!(
        missing_swift_rust_targets(
            "aarch64-apple-darwin\nx86_64-apple-darwin\naarch64-apple-ios\nx86_64-apple-ios\n"
        ),
        vec!["aarch64-apple-ios-sim"]
    );
}

#[test]
fn backend_labels_used_by_doctor_are_cli_labels() {
    assert_eq!(Backend::All.as_str(), "all");
    assert_eq!(Backend::Vulkan.as_str(), "vulkan");
}
