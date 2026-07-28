//! Tests the `targets::swift` module in `xtask`.
//!
//! Covers deterministic generation, source staging, architecture, ABI,
//! linkage, path construction, and platform gates without Apple toolchains.

use std::collections::BTreeSet;

use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{
    copy_dir_recursive, declared_ffi_symbols, ensure_macos_host, ensure_swift_sources,
    exported_ffi_symbols, validate_architecture_list, validate_dynamic_linkage_output,
    validate_ffi_symbol_sets, validate_identical_directories, validate_nm_diagnostics,
    validate_sandbox_entitlements, AppleSlice, SliceBackend, IOS_ARM64, IOS_SIMULATOR_ARM64,
    IOS_SIMULATOR_X86_64, MACOS_ARM64, MACOS_X86_64,
};

#[test]
fn apple_slice_policy_is_explicit() {
    assert_eq!(
        [
            MACOS_ARM64,
            MACOS_X86_64,
            IOS_ARM64,
            IOS_SIMULATOR_ARM64,
            IOS_SIMULATOR_X86_64,
        ],
        [
            AppleSlice {
                target: "aarch64-apple-darwin",
                architecture: "arm64",
                backend: SliceBackend::Metal,
                deployment_target: "11.0",
            },
            AppleSlice {
                target: "x86_64-apple-darwin",
                architecture: "x86_64",
                backend: SliceBackend::Cpu,
                deployment_target: "10.15",
            },
            AppleSlice {
                target: "aarch64-apple-ios",
                architecture: "arm64",
                backend: SliceBackend::Metal,
                deployment_target: "16.0",
            },
            AppleSlice {
                target: "aarch64-apple-ios-sim",
                architecture: "arm64",
                backend: SliceBackend::Cpu,
                deployment_target: "16.0",
            },
            AppleSlice {
                target: "x86_64-apple-ios",
                architecture: "x86_64",
                backend: SliceBackend::Cpu,
                deployment_target: "16.0",
            },
        ]
    );
}

#[test]
fn recursive_copy_preserves_package_tree() {
    let temp = TempDir::new("target-swift-copy");
    let source = temp.create_dir("source");
    temp.write("source/Sipp.swift", "public enum Sipp {}\n");
    temp.write("source/Nested/Test.swift", "import XCTest\n");
    let destination = temp.join("destination");

    copy_dir_recursive(&source, &destination).unwrap();

    assert_eq!(
        std::fs::read_to_string(destination.join("Sipp.swift")).unwrap(),
        "public enum Sipp {}\n"
    );
    assert!(destination.join("Nested/Test.swift").is_file());
}

#[test]
fn source_validation_requires_binding_package_examples_and_submodule() {
    let temp = TempDir::new("target-swift-inputs");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    let error = ensure_swift_sources(&ctx).unwrap_err();
    assert!(error.to_string().contains("bindings"));

    temp.write("bindings/swift/Cargo.toml", "");
    temp.write("bindings/swift/uniffi.toml", "");
    temp.write("bindings/swift/uniffi-bindgen.toml", "");
    temp.write("lib/swift/Package.swift", "");
    temp.write("lib/swift/Support/module.modulemap", "");
    temp.write("lib/swift/Consumer/Package.swift", "");
    temp.write("lib/swift/Consumer/Sources/SippConsumer/Consumer.swift", "");
    temp.write("examples/swift/README.md", "");
    temp.write("examples/swift/cli/Package.swift", "");
    temp.write("examples/swift/cli/Sources/SippCLI/SippCLI.swift", "");
    temp.write("examples/swift/swiftui/Package.swift", "");
    temp.write("examples/swift/swiftui/Info.plist", "");
    temp.write("examples/swift/swiftui/SippSandbox.entitlements", "");
    temp.write(
        "examples/swift/swiftui/Sources/SippSandbox/SippSandboxApp.swift",
        "",
    );
    temp.write(
        "examples/swift/swiftui/Sources/SippSandbox/ContentView.swift",
        "",
    );
    temp.write(
        "examples/swift/swiftui/Sources/SippSandbox/SippViewModel.swift",
        "",
    );
    temp.write("examples/swift/ios/SippIOS.xcodeproj/project.pbxproj", "");
    temp.write("examples/swift/ios/SippIOSApp.swift", "");
    temp.write("examples/swift/ios/Info.plist", "");
    temp.write("crates/sys/llama.cpp/include/llama.h", "");

    ensure_swift_sources(&ctx).unwrap();
}

#[test]
fn host_gate_matches_the_current_operating_system() {
    if cfg!(target_os = "macos") {
        ensure_macos_host().unwrap();
    } else {
        let error = ensure_macos_host().unwrap_err();
        assert!(error.to_string().contains("require macOS"));
    }
}

#[test]
fn deterministic_generation_requires_identical_paths_and_bytes() {
    let temp = TempDir::new("target-swift-determinism");
    let first = temp.create_dir("first");
    let second = temp.create_dir("second");
    temp.write("first/SippCoreBindings.swift", "generated\n");
    temp.write("second/SippCoreBindings.swift", "generated\n");

    validate_identical_directories(&first, &second).unwrap();

    temp.write("second/SippCoreBindings.swift", "changed\n");
    assert!(validate_identical_directories(&first, &second).is_err());
}

#[test]
fn architecture_validation_requires_the_exact_slice_set() {
    validate_architecture_list("x86_64 arm64\n", &["arm64", "x86_64"]).unwrap();
    assert!(validate_architecture_list("arm64\n", &["arm64", "x86_64"]).is_err());
    assert!(validate_architecture_list("arm64 x86_64 i386\n", &["arm64", "x86_64"]).is_err());
}

#[test]
fn abi_validation_matches_generated_declarations_to_archive_exports() {
    let header = "\
uint32_t ffi_sipp_swift_uniffi_contract_version(void);
uint16_t uniffi_sipp_swift_checksum_method_ffi_sipp_client_models(void);
";
    let declared = declared_ffi_symbols(header).unwrap();
    let exported = exported_ffi_symbols(
        "_ffi_sipp_swift_uniffi_contract_version\n_uniffi_sipp_swift_checksum_method_ffi_sipp_client_models\n",
    );

    validate_ffi_symbol_sets(&declared, &exported).unwrap();

    let incomplete_exports = BTreeSet::from(["ffi_sipp_swift_uniffi_contract_version".to_owned()]);
    assert!(validate_ffi_symbol_sets(&declared, &incomplete_exports).is_err());

    let mut leaked_exports = exported;
    leaked_exports.insert("ffi_sipp_swift_undeclared".to_owned());
    assert!(validate_ffi_symbol_sets(&declared, &leaked_exports).is_err());
}

#[test]
fn nm_validation_accepts_only_the_known_apple_llvm_version_mismatch() {
    assert!(!validate_nm_diagnostics(true, "").unwrap());
    assert!(validate_nm_diagnostics(
        false,
        "nm: error: object: Unknown attribute kind (102) (Producer: 'LLVM22' Reader: 'LLVM APPLE_1')"
    )
    .unwrap());
    assert!(validate_nm_diagnostics(false, "nm: error: archive is corrupt").is_err());
    assert!(validate_nm_diagnostics(false, "nm: no symbols").is_err());
}

#[test]
fn linkage_validation_accepts_only_system_and_swift_runtime_dependencies() {
    let valid = "\
/tmp/SippConsumer:
    /System/Library/Frameworks/Foundation.framework/Versions/C/Foundation (compatibility version 300.0.0)
    /usr/lib/libc++.1.dylib (compatibility version 1.0.0)
    @rpath/libswiftCore.dylib (compatibility version 1.0.0)
";
    validate_dynamic_linkage_output(valid).unwrap();

    let sidecar = "\
/tmp/SippConsumer:
    @rpath/libsipp_runtime.dylib (compatibility version 1.0.0)
";
    assert!(validate_dynamic_linkage_output(sidecar).is_err());
}

#[test]
fn sandbox_entitlements_are_exact() {
    let expected = r#"{
        "com.apple.security.app-sandbox": true,
        "com.apple.security.files.bookmarks.app-scope": true,
        "com.apple.security.files.user-selected.read-only": true
    }"#;
    validate_sandbox_entitlements(expected).unwrap();

    let leaked = r#"{
        "com.apple.security.app-sandbox": true,
        "com.apple.security.files.bookmarks.app-scope": true,
        "com.apple.security.files.user-selected.read-only": true,
        "com.apple.security.network.client": true
    }"#;
    assert!(validate_sandbox_entitlements(leaked).is_err());
    assert!(validate_sandbox_entitlements(r#"{"com.apple.security.app-sandbox": false}"#).is_err());
}
