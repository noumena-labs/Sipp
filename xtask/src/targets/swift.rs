//! Swift binding and XCFramework build target.

use crate::cli::Backend;
use crate::output;
use crate::toolchains::env::apply_toolchains;
use crate::utils::{sha256_file, BuildContext};
use anyhow::{Context, Result};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;
use xshell::{cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////
#[cfg(test)]
#[path = "../tests/targets/swift_tests.rs"]
mod swift_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
const SWIFT_BINDING_PACKAGE: &str = "sipp-swift";
const SWIFT_BINDGEN_PACKAGE: &str = "sipp-uniffi-bindgen-swift";
const SWIFT_LIBRARY_NAME: &str = "libsipp_swift.a";
const SWIFT_MODULE_NAME: &str = "SippCoreBindings";
const SWIFT_HEADER_NAME: &str = "SippCoreFFI.h";
const SWIFT_XCFRAMEWORK_NAME: &str = "SippCore.xcframework";
const SWIFT_DISTRIBUTION_ARCHIVE: &str = "SippCore.xcframework.zip";
const SWIFT_DISTRIBUTION_CHECKSUM: &str = "SippCore.xcframework.zip.sha256";
const SWIFT_ABI_MANIFEST: &str = "SippCore.abi.txt";
const SWIFT_PACKAGE_SCHEME: &str = "Sipp";
const SWIFT_CLI_NAME: &str = "SippCLI";
const SWIFT_SANDBOX_NAME: &str = "SippSandbox";
const SWIFT_IOS_APP_NAME: &str = "SippIOS";
/// Bundle identifier used by the iOS example project and simulator workflow.
pub(crate) const SWIFT_IOS_APP_BUNDLE_ID: &str = "ai.sipp.examples.ios";
const SWIFT_MACOS_DEPLOYMENT_TARGET: &str = "11.0";
const SWIFT_IOS_DEPLOYMENT_TARGET: &str = "16.0";
const SWIFT_PACKAGE_DESTINATIONS: [(&str, &str); 3] = [
    ("macos", "generic/platform=macOS"),
    ("ios", "generic/platform=iOS"),
    ("ios-simulator", "generic/platform=iOS Simulator"),
];
pub(crate) const XCRUN_PATH: &str = "/usr/bin/xcrun";
pub(crate) const DITTO_PATH: &str = "/usr/bin/ditto";
pub(crate) const CODESIGN_PATH: &str = "/usr/bin/codesign";
pub(crate) const PLUTIL_PATH: &str = "/usr/bin/plutil";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SliceBackend {
    Cpu,
    Metal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct AppleSlice {
    target: &'static str,
    architecture: &'static str,
    backend: SliceBackend,
    deployment_target: &'static str,
}

const MACOS_ARM64: AppleSlice = AppleSlice {
    target: "aarch64-apple-darwin",
    architecture: "arm64",
    backend: SliceBackend::Metal,
    deployment_target: SWIFT_MACOS_DEPLOYMENT_TARGET,
};
const MACOS_X86_64: AppleSlice = AppleSlice {
    target: "x86_64-apple-darwin",
    architecture: "x86_64",
    backend: SliceBackend::Cpu,
    deployment_target: "10.15",
};
const IOS_ARM64: AppleSlice = AppleSlice {
    target: "aarch64-apple-ios",
    architecture: "arm64",
    backend: SliceBackend::Metal,
    deployment_target: SWIFT_IOS_DEPLOYMENT_TARGET,
};
const IOS_SIMULATOR_ARM64: AppleSlice = AppleSlice {
    target: "aarch64-apple-ios-sim",
    architecture: "arm64",
    backend: SliceBackend::Cpu,
    deployment_target: SWIFT_IOS_DEPLOYMENT_TARGET,
};
const IOS_SIMULATOR_X86_64: AppleSlice = AppleSlice {
    target: "x86_64-apple-ios",
    architecture: "x86_64",
    backend: SliceBackend::Cpu,
    deployment_target: SWIFT_IOS_DEPLOYMENT_TARGET,
};

/// Builds and validates the distributable Apple Swift package.
pub fn build(sh: &Shell, ctx: &BuildContext, examples: bool) -> Result<()> {
    ensure_macos_host()?;
    ensure_swift_sources(ctx)?;
    if examples {
        ensure_swift_example_sources(ctx)?;
    }

    let started_at = Instant::now();
    let artifacts_dir = ctx.swift_artifacts_dir();
    output::phase("Swift distribution");
    output::path("Binding workspace", &ctx.bindings_swift_dir());
    output::path("Package source", &ctx.swift_package_dir());
    output::path("Artifact directory", &artifacts_dir);

    let staging_dir = ctx.tmp_dir().join("swift");
    prepare_output(sh, &staging_dir, &artifacts_dir)?;

    let target_dir = ctx.cargo_swift_target_dir();
    let macos_arm64 = build_archive(sh, ctx, &target_dir, MACOS_ARM64)?;
    let macos_x86_64 = build_archive(sh, ctx, &target_dir, MACOS_X86_64)?;
    let ios_arm64 = build_archive(sh, ctx, &target_dir, IOS_ARM64)?;
    let ios_simulator_arm64 = build_archive(sh, ctx, &target_dir, IOS_SIMULATOR_ARM64)?;
    let ios_simulator_x86_64 = build_archive(sh, ctx, &target_dir, IOS_SIMULATOR_X86_64)?;
    let generated_dir = staging_dir.join("generated");
    generate_bindings(sh, ctx, &macos_arm64, &generated_dir)?;

    let macos_archive =
        create_universal_archive(sh, &staging_dir.join("macos"), &macos_arm64, &macos_x86_64)?;
    let ios_simulator_archive = create_universal_archive(
        sh,
        &staging_dir.join("ios-simulator"),
        &ios_simulator_arm64,
        &ios_simulator_x86_64,
    )?;
    validate_archive_architectures(sh, &macos_archive, &["arm64", "x86_64"])?;
    validate_archive_architectures(sh, &ios_simulator_archive, &["arm64", "x86_64"])?;
    let header = generated_dir.join("Headers").join(SWIFT_HEADER_NAME);
    let header_contents = fs::read_to_string(&header)
        .with_context(|| format!("failed to read generated header {}", header.display()))?;
    let declared_symbols = declared_ffi_symbols(&header_contents)?;
    for archive in [&macos_archive, &ios_arm64, &ios_simulator_archive] {
        validate_ffi_exports(archive, &declared_symbols)?;
    }
    let abi_manifest = artifacts_dir.join(SWIFT_ABI_MANIFEST);
    fs::write(
        &abi_manifest,
        declared_symbols
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .with_context(|| format!("failed to write ABI manifest {}", abi_manifest.display()))?;
    let package_dir = stage_package(sh, ctx, &generated_dir)?;
    let xcframework = package_dir.join("Binary").join(SWIFT_XCFRAMEWORK_NAME);
    let headers_dir = generated_dir.join("Headers");
    create_xcframework(
        sh,
        &xcframework,
        &headers_dir,
        &[&macos_archive, &ios_arm64, &ios_simulator_archive],
    )?;
    validate_package_platforms(sh, ctx, &package_dir)?;

    let (distribution_archive, distribution_checksum) =
        archive_distribution(sh, ctx, &xcframework)?;

    if examples {
        build_examples(sh, ctx)?;
    }

    output::artifact(&xcframework);
    output::artifact(&abi_manifest);
    output::artifact(&distribution_archive);
    output::artifact(&distribution_checksum);
    output::success(format!(
        "Swift build complete in {}",
        output::elapsed(started_at.elapsed())
    ));
    Ok(())
}

/// Builds the host macOS slice and runs the staged Swift package unit tests.
pub fn test(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    ensure_macos_host()?;
    ensure_swift_sources(ctx)?;

    let started_at = Instant::now();
    let artifacts_dir = ctx.swift_artifacts_dir();
    output::phase("Swift package tests");
    output::path("Binding workspace", &ctx.bindings_swift_dir());
    output::path("Package source", &ctx.swift_package_dir());
    output::path("Artifact directory", &artifacts_dir);

    let staging_dir = ctx.tmp_dir().join("swift");
    prepare_output(sh, &staging_dir, &artifacts_dir)?;

    let target_dir = ctx.cargo_swift_target_dir();
    let host_slice = macos_slice_for_architecture(std::env::consts::ARCH)?;
    let host_archive = build_archive(sh, ctx, &target_dir, host_slice)?;
    let generated_dir = staging_dir.join("generated");
    generate_bindings(sh, ctx, &host_archive, &generated_dir)?;
    let package_dir = stage_package(sh, ctx, &generated_dir)?;
    let xcframework = package_dir.join("Binary").join(SWIFT_XCFRAMEWORK_NAME);
    create_xcframework(
        sh,
        &xcframework,
        &generated_dir.join("Headers"),
        &[&host_archive],
    )?;

    let _dir = sh.push_dir(&package_dir);
    output::run_test_command(
        "Running staged Swift package tests",
        cmd!(sh, "{XCRUN_PATH} swift test"),
    )
    .context("staged Swift package tests failed")?;

    output::success(format!(
        "Swift package tests complete in {}",
        output::elapsed(started_at.elapsed())
    ));
    Ok(())
}

fn macos_slice_for_architecture(architecture: &str) -> Result<AppleSlice> {
    match architecture {
        "aarch64" => Ok(MACOS_ARM64),
        "x86_64" => Ok(MACOS_X86_64),
        unsupported => anyhow::bail!(
            "Swift package tests do not support macOS host architecture {unsupported}"
        ),
    }
}

fn ensure_macos_host() -> Result<()> {
    if !cfg!(target_os = "macos") {
        anyhow::bail!(
            "Swift XCFramework builds require macOS with Xcode; run `cargo xtask build swift` on a macOS host"
        );
    }
    Ok(())
}

fn ensure_swift_sources(ctx: &BuildContext) -> Result<()> {
    let required = [
        ctx.bindings_swift_dir().join("Cargo.toml"),
        ctx.bindings_swift_dir().join("uniffi.toml"),
        ctx.bindings_swift_dir().join("uniffi-bindgen.toml"),
        ctx.swift_package_dir().join("Package.swift"),
        ctx.swift_package_dir().join("Support/module.modulemap"),
        ctx.llama_cpp_dir().join("include/llama.h"),
    ];
    validate_required_sources(&required, "Swift build")
}

fn ensure_swift_example_sources(ctx: &BuildContext) -> Result<()> {
    let required = [
        ctx.workspace_root().join("examples/swift/README.md"),
        ctx.workspace_root()
            .join("examples/swift/cli/Package.swift"),
        ctx.workspace_root()
            .join("examples/swift/cli/Sources/SippCLI/SippCLI.swift"),
        ctx.workspace_root()
            .join("examples/swift/swiftui/Package.swift"),
        ctx.workspace_root()
            .join("examples/swift/swiftui/Info.plist"),
        ctx.workspace_root()
            .join("examples/swift/swiftui/SippSandbox.entitlements"),
        ctx.workspace_root()
            .join("examples/swift/swiftui/Sources/SippSandbox/SippSandboxApp.swift"),
        ctx.workspace_root()
            .join("examples/swift/swiftui/Sources/SippSandbox/ContentView.swift"),
        ctx.workspace_root()
            .join("examples/swift/swiftui/Sources/SippSandbox/SippViewModel.swift"),
        ctx.workspace_root()
            .join("examples/swift/ios/SippIOS.xcodeproj/project.pbxproj"),
        ctx.workspace_root()
            .join("examples/swift/ios/SippIOSApp.swift"),
        ctx.workspace_root().join("examples/swift/ios/Info.plist"),
    ];
    validate_required_sources(&required, "Swift example")
}

fn validate_required_sources(required: &[PathBuf], label: &str) -> Result<()> {
    for path in required {
        if !path.is_file() {
            anyhow::bail!("required {label} input is missing: {}", path.display());
        }
    }
    Ok(())
}

fn prepare_output(sh: &Shell, staging_dir: &Path, artifacts_dir: &Path) -> Result<()> {
    for path in [staging_dir, artifacts_dir] {
        if path.exists() {
            sh.remove_path(path)
                .with_context(|| format!("failed to remove {}", path.display()))?;
        }
        sh.create_dir(path)
            .with_context(|| format!("failed to create {}", path.display()))?;
    }
    Ok(())
}

fn build_archive(
    sh: &Shell,
    ctx: &BuildContext,
    target_dir: &Path,
    slice: AppleSlice,
) -> Result<PathBuf> {
    let target = slice.target;
    output::phase(&format!("Swift native slice: {target}"));
    output::detail("Deployment target", slice.deployment_target);
    let cmake_dir = ctx.cmake_swift_sys_dir(target);
    sh.create_dir(&cmake_dir)?;
    let cmake_dir = cmake_dir
        .canonicalize()
        .with_context(|| format!("failed to canonicalize {}", cmake_dir.display()))?;
    let (cargo_cmd, native_backend) = match slice.backend {
        SliceBackend::Cpu => (cmd!(
            sh,
            "cargo build --locked --release --package {SWIFT_BINDING_PACKAGE} --target {target} --target-dir {target_dir}"
        ), Backend::Cpu),
        SliceBackend::Metal => (cmd!(
            sh,
            "cargo build --locked --release --package {SWIFT_BINDING_PACKAGE} --features metal --target {target} --target-dir {target_dir}"
        ), Backend::Metal),
    };
    let cargo_cmd = apply_toolchains(sh, ctx, cargo_cmd, Some(&native_backend))?
        .env("SIPP_SYS_CMAKE_OUT_DIR", &cmake_dir)
        // Apple LLVM tools must inspect the archive when packaging it. Fat
        // Rust LTO can leave newer LLVM bitcode that older supported Xcode
        // releases cannot read, even though the native slice links correctly.
        .env("CARGO_PROFILE_RELEASE_LTO", "false");
    let cargo_cmd = if target.ends_with("apple-darwin") {
        cargo_cmd.env("MACOSX_DEPLOYMENT_TARGET", slice.deployment_target)
    } else {
        cargo_cmd.env("IPHONEOS_DEPLOYMENT_TARGET", slice.deployment_target)
    };
    output::run_build_command(format!("Compiling Swift native slice {target}"), cargo_cmd)
        .with_context(|| format!("failed to compile Swift native slice {target}"))?;

    let archive = target_dir
        .join(target)
        .join("release")
        .join(SWIFT_LIBRARY_NAME);
    if !archive.is_file() {
        anyhow::bail!(
            "Swift static archive was not produced for {target}: {}",
            archive.display()
        );
    }
    validate_archive_architectures(sh, &archive, &[slice.architecture])?;
    output::artifact(&archive);
    Ok(archive)
}

fn generate_bindings(
    sh: &Shell,
    ctx: &BuildContext,
    metadata_archive: &Path,
    generated_dir: &Path,
) -> Result<()> {
    let headers_dir = generated_dir.join("Headers");
    sh.create_dir(generated_dir)?;
    sh.create_dir(&headers_dir)?;

    let bindgen_target_dir = ctx.cargo_build_root().join("swift-bindgen");
    let bindgen_config = ctx.bindings_swift_dir().join("uniffi-bindgen.toml");
    let bindgen_cmd = cmd!(
        sh,
        "cargo run --locked --release --package {SWIFT_BINDGEN_PACKAGE} --target-dir {bindgen_target_dir} -- --config {bindgen_config} {metadata_archive} {generated_dir} --swift-sources --metadata-no-deps"
    );
    output::run_build_command("Generating Swift UniFFI source", bindgen_cmd)
        .context("failed to generate Swift UniFFI source")?;

    let header_cmd = cmd!(
        sh,
        "cargo run --locked --release --package {SWIFT_BINDGEN_PACKAGE} --target-dir {bindgen_target_dir} -- --config {bindgen_config} {metadata_archive} {headers_dir} --headers --metadata-no-deps"
    );
    output::run_build_command("Generating Swift UniFFI header", header_cmd)
        .context("failed to generate Swift UniFFI header")?;

    let swift_source = generated_dir.join(format!("{SWIFT_MODULE_NAME}.swift"));
    let header = headers_dir.join(SWIFT_HEADER_NAME);
    for path in [&swift_source, &header] {
        if !path.is_file() {
            anyhow::bail!("UniFFI did not generate expected file {}", path.display());
        }
    }

    fs::copy(
        ctx.swift_package_dir().join("Support/module.modulemap"),
        headers_dir.join("module.modulemap"),
    )
    .context("failed to stage the explicit Swift module map")?;
    Ok(())
}

fn create_universal_archive(
    sh: &Shell,
    output_dir: &Path,
    arm64: &Path,
    x86_64: &Path,
) -> Result<PathBuf> {
    sh.create_dir(output_dir)?;
    let output_archive = output_dir.join(SWIFT_LIBRARY_NAME);
    output::run_build_command(
        "Creating universal Swift archive",
        cmd!(
            sh,
            "{XCRUN_PATH} lipo -create {arm64} {x86_64} -output {output_archive}"
        ),
    )
    .context("failed to create universal Swift archive")?;
    Ok(output_archive)
}

fn create_xcframework(
    sh: &Shell,
    xcframework: &Path,
    headers_dir: &Path,
    archives: &[&Path],
) -> Result<()> {
    let mut command = cmd!(sh, "{XCRUN_PATH} xcodebuild -create-xcframework");
    for archive in archives {
        command = command
            .arg("-library")
            .arg(archive)
            .arg("-headers")
            .arg(headers_dir);
    }
    command = command.arg("-output").arg(xcframework);
    output::run_build_command("Creating SippCore.xcframework", command)
        .context("failed to create SippCore.xcframework")
}

fn stage_package(sh: &Shell, ctx: &BuildContext, generated_dir: &Path) -> Result<PathBuf> {
    let source = ctx.swift_package_dir();
    let destination = ctx.swift_package_artifacts_dir();
    sh.create_dir(&destination)?;

    fs::copy(
        source.join("Package.swift"),
        destination.join("Package.swift"),
    )
    .context("failed to stage Swift Package.swift")?;
    fs::copy(source.join("README.md"), destination.join("README.md"))
        .context("failed to stage Swift README")?;
    copy_dir_recursive(&source.join("Sources"), &destination.join("Sources"))?;
    copy_dir_recursive(&source.join("Tests"), &destination.join("Tests"))?;

    let generated_source = generated_dir.join(format!("{SWIFT_MODULE_NAME}.swift"));
    let generated_directory = destination
        .join("Sources")
        .join("SippCoreBindings")
        .join("Generated");
    fs::create_dir_all(&generated_directory)
        .with_context(|| format!("failed to create {}", generated_directory.display()))?;
    let generated_destination = generated_directory.join(format!("{SWIFT_MODULE_NAME}.swift"));
    fs::copy(&generated_source, &generated_destination).with_context(|| {
        format!(
            "failed to copy generated Swift source from {} to {}",
            generated_source.display(),
            generated_destination.display()
        )
    })?;

    sh.create_dir(destination.join("Binary"))?;
    Ok(destination)
}

fn copy_dir_recursive(source: &Path, destination: &Path) -> Result<()> {
    fs::create_dir_all(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;
    for entry in
        fs::read_dir(source).with_context(|| format!("failed to read {}", source.display()))?
    {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&source_path, &destination_path)?;
        } else {
            fs::copy(&source_path, &destination_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    source_path.display(),
                    destination_path.display()
                )
            })?;
        }
    }
    Ok(())
}

fn validate_archive_architectures(sh: &Shell, archive: &Path, expected: &[&str]) -> Result<()> {
    let architectures = cmd!(sh, "{XCRUN_PATH} lipo -archs {archive}")
        .read()
        .with_context(|| format!("failed to inspect architectures in {}", archive.display()))?;
    validate_architecture_list(&architectures, expected)?;
    output::success(format!(
        "Validated {} architecture(s): {}",
        archive.display(),
        expected.join(", ")
    ));
    Ok(())
}

fn validate_architecture_list(actual: &str, expected: &[&str]) -> Result<()> {
    let actual = actual.split_whitespace().collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        anyhow::bail!(
            "Swift archive architectures were [{}]; expected [{}]",
            actual.into_iter().collect::<Vec<_>>().join(", "),
            expected.into_iter().collect::<Vec<_>>().join(", ")
        );
    }
    Ok(())
}

fn validate_ffi_exports(archive: &Path, declared: &BTreeSet<String>) -> Result<()> {
    let nm_output = Command::new(XCRUN_PATH)
        .args(["nm", "-gjU"])
        .arg(archive)
        .output()
        .with_context(|| {
            format!(
                "failed to inspect exported symbols in {}",
                archive.display()
            )
        })?;
    let exported = exported_ffi_symbols(&String::from_utf8_lossy(&nm_output.stdout));
    validate_ffi_symbol_sets(declared, &exported)?;
    if validate_nm_diagnostics(
        nm_output.status.success(),
        &String::from_utf8_lossy(&nm_output.stderr),
    )? {
        output::warning(
            "Apple nm skipped incompatible LLVM metadata after reading all expected Sipp FFI exports",
        );
    }

    output::success(format!(
        "Validated {} Swift FFI exports in {}",
        declared.len(),
        archive.display()
    ));
    Ok(())
}

fn validate_nm_diagnostics(success: bool, stderr: &str) -> Result<bool> {
    if success {
        return Ok(false);
    }

    let errors = stderr.lines().filter(|line| line.contains("error:"));
    let mut saw_llvm_version_mismatch = false;
    for error in errors {
        let known_mismatch = (error.contains("Unknown attribute kind")
            || error.contains("Invalid attribute group entry"))
            && error.contains("Producer: 'LLVM")
            && error.contains("Reader: 'LLVM APPLE_");
        if !known_mismatch {
            anyhow::bail!("Apple nm failed: {error}");
        }
        saw_llvm_version_mismatch = true;
    }
    if !saw_llvm_version_mismatch {
        anyhow::bail!("Apple nm failed without a recognized LLVM compatibility diagnostic");
    }
    Ok(true)
}

fn declared_ffi_symbols(header: &str) -> Result<BTreeSet<String>> {
    let symbols = header
        .lines()
        .filter_map(|line| line.split_once('(').map(|(prefix, _)| prefix))
        .filter_map(|prefix| prefix.split_whitespace().last())
        .map(|symbol| symbol.trim_start_matches('*'))
        .filter(|symbol| {
            symbol.starts_with("ffi_sipp_swift_") || symbol.starts_with("uniffi_sipp_swift_")
        })
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if symbols.is_empty() {
        anyhow::bail!("generated Swift header declares no Sipp FFI symbols");
    }
    Ok(symbols)
}

fn exported_ffi_symbols(nm_output: &str) -> BTreeSet<String> {
    nm_output
        .lines()
        .filter_map(|line| line.split_whitespace().last())
        .map(|symbol| symbol.trim_start_matches('_'))
        .filter(|symbol| {
            symbol.starts_with("ffi_sipp_swift_") || symbol.starts_with("uniffi_sipp_swift_")
        })
        .map(str::to_owned)
        .collect()
}

fn validate_ffi_symbol_sets(
    declared: &BTreeSet<String>,
    exported: &BTreeSet<String>,
) -> Result<()> {
    let missing = declared.difference(exported).cloned().collect::<Vec<_>>();
    if !missing.is_empty() {
        anyhow::bail!(
            "Swift archive is missing generated FFI exports: {}",
            missing.join(", ")
        );
    }
    let unexpected = exported.difference(declared).cloned().collect::<Vec<_>>();
    if !unexpected.is_empty() {
        anyhow::bail!(
            "Swift archive exports undeclared Sipp FFI symbols: {}",
            unexpected.join(", ")
        );
    }
    Ok(())
}

fn validate_package_platforms(sh: &Shell, ctx: &BuildContext, package_dir: &Path) -> Result<()> {
    let validation_dir = ctx.swift_artifacts_dir().join("platform-validation");
    for (name, destination) in SWIFT_PACKAGE_DESTINATIONS {
        let derived_data = validation_dir.join(name);
        let _dir = sh.push_dir(package_dir);
        output::run_build_command(
            format!("Building staged Swift package for {name}"),
            cmd!(
                sh,
                "{XCRUN_PATH} xcodebuild -quiet -scheme {SWIFT_PACKAGE_SCHEME} -configuration Release -destination {destination} -derivedDataPath {derived_data} CODE_SIGNING_ALLOWED=NO build"
            ),
        )
        .with_context(|| format!("staged Swift package failed to build for {name}"))?;
    }
    Ok(())
}

fn build_examples(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let examples_source = ctx.workspace_root().join("examples/swift");
    let examples_dir = ctx.swift_artifacts_dir().join("examples");
    let cli_dir = examples_dir.join("cli");
    copy_dir_recursive(&examples_source.join("cli"), &cli_dir)?;
    let built_cli = build_swift_executable_package(sh, &cli_dir, SWIFT_CLI_NAME)?;
    let cli_executable = examples_dir.join(SWIFT_CLI_NAME);
    fs::copy(&built_cli, &cli_executable).with_context(|| {
        format!(
            "failed to stage Swift CLI from {} to {}",
            built_cli.display(),
            cli_executable.display()
        )
    })?;
    validate_dynamic_linkage(sh, &cli_executable)?;

    let swiftui_dir = examples_dir.join("swiftui");
    copy_dir_recursive(&examples_source.join("swiftui"), &swiftui_dir)?;
    let swiftui_executable = build_swift_executable_package(sh, &swiftui_dir, SWIFT_SANDBOX_NAME)?;
    validate_dynamic_linkage(sh, &swiftui_executable)?;
    let sandbox_app =
        bundle_and_sign_sandbox_app(sh, &swiftui_dir, &swiftui_executable, &examples_dir)?;
    let ios_app = build_ios_example(sh, ctx)?;

    output::artifact(&cli_executable);
    output::artifact(&sandbox_app);
    output::artifact(&ios_app);
    Ok(())
}

fn build_swift_executable_package(
    sh: &Shell,
    package_dir: &Path,
    executable_name: &str,
) -> Result<PathBuf> {
    let _dir = sh.push_dir(package_dir);
    output::run_build_command(
        format!("Building {executable_name} Swift package"),
        cmd!(sh, "{XCRUN_PATH} swift build -c release"),
    )
    .with_context(|| format!("{executable_name} Swift package failed to build"))?;
    let bin_path = cmd!(sh, "{XCRUN_PATH} swift build -c release --show-bin-path")
        .read()
        .with_context(|| format!("failed to resolve {executable_name} binary path"))?;
    let executable = PathBuf::from(bin_path.trim()).join(executable_name);
    if !executable.is_file() {
        anyhow::bail!(
            "{executable_name} Swift executable was not produced: {}",
            executable.display()
        );
    }
    Ok(executable)
}

/// Returns the iOS Simulator app produced by the Swift build.
pub(crate) fn ios_example_app(ctx: &BuildContext) -> PathBuf {
    ios_example_derived_data(ctx)
        .join("Build/Products/Debug-iphonesimulator")
        .join(format!("{SWIFT_IOS_APP_NAME}.app"))
}

fn ios_example_derived_data(ctx: &BuildContext) -> PathBuf {
    ctx.swift_artifacts_dir().join("examples/ios-derived")
}

fn build_ios_example(sh: &Shell, ctx: &BuildContext) -> Result<PathBuf> {
    let project = ctx
        .workspace_root()
        .join("examples/swift/ios/SippIOS.xcodeproj");
    let derived_data = ios_example_derived_data(ctx);
    let destination = "generic/platform=iOS Simulator";
    output::run_build_command(
        "Building Sipp iOS example",
        cmd!(
            sh,
            "{XCRUN_PATH} xcodebuild -quiet -project {project} -scheme {SWIFT_IOS_APP_NAME} -configuration Debug -destination {destination} -derivedDataPath {derived_data} CODE_SIGNING_ALLOWED=NO build"
        ),
    )
    .context("Sipp iOS example failed to build")?;

    let app = ios_example_app(ctx);
    if !app.is_dir() {
        anyhow::bail!("Sipp iOS app was not produced: {}", app.display());
    }
    Ok(app)
}

fn bundle_and_sign_sandbox_app(
    sh: &Shell,
    package_dir: &Path,
    executable: &Path,
    examples_dir: &Path,
) -> Result<PathBuf> {
    let app = examples_dir.join(format!("{SWIFT_SANDBOX_NAME}.app"));
    let contents = app.join("Contents");
    let macos = contents.join("MacOS");
    fs::create_dir_all(&macos).with_context(|| format!("failed to create {}", macos.display()))?;
    fs::copy(executable, macos.join(SWIFT_SANDBOX_NAME)).with_context(|| {
        format!(
            "failed to stage SwiftUI executable from {}",
            executable.display()
        )
    })?;
    fs::copy(package_dir.join("Info.plist"), contents.join("Info.plist"))
        .context("failed to stage SwiftUI Info.plist")?;

    let entitlements = package_dir.join("SippSandbox.entitlements");
    let entitlement_json = cmd!(sh, "{PLUTIL_PATH} -convert json -o - {entitlements}")
        .read()
        .with_context(|| format!("failed to read {}", entitlements.display()))?;
    validate_sandbox_entitlements(&entitlement_json)?;
    output::run_build_command(
        "Signing sandboxed SwiftUI example",
        cmd!(
            sh,
            "{CODESIGN_PATH} --force --sign - --entitlements {entitlements} {app}"
        ),
    )
    .context("failed to sign sandboxed SwiftUI example")?;
    output::run_build_command(
        "Verifying sandboxed SwiftUI signature",
        cmd!(sh, "{CODESIGN_PATH} --verify --strict --verbose=2 {app}"),
    )
    .context("sandboxed SwiftUI signature verification failed")?;
    Ok(app)
}

fn validate_sandbox_entitlements(contents: &str) -> Result<()> {
    let actual = serde_json::from_str::<BTreeMap<String, bool>>(contents)
        .context("sandbox entitlements must be a boolean dictionary")?;
    let expected = BTreeMap::from([
        ("com.apple.security.app-sandbox".to_owned(), true),
        (
            "com.apple.security.files.bookmarks.app-scope".to_owned(),
            true,
        ),
        (
            "com.apple.security.files.user-selected.read-only".to_owned(),
            true,
        ),
    ]);
    if actual != expected {
        anyhow::bail!(
            "sandbox entitlements were {}; expected {}",
            serde_json::to_string(&actual)?,
            serde_json::to_string(&expected)?
        );
    }
    Ok(())
}

fn validate_dynamic_linkage(sh: &Shell, executable: &Path) -> Result<()> {
    let otool_output = cmd!(sh, "{XCRUN_PATH} otool -L {executable}")
        .read()
        .with_context(|| format!("failed to inspect linkage for {}", executable.display()))?;
    validate_dynamic_linkage_output(&otool_output)?;
    output::success(format!(
        "Validated {} dynamic linkage",
        executable.display()
    ));
    Ok(())
}

fn validate_dynamic_linkage_output(otool_output: &str) -> Result<()> {
    let dependencies = otool_output
        .lines()
        .skip(1)
        .filter_map(|line| line.split_whitespace().next())
        .collect::<Vec<_>>();
    if dependencies.is_empty() {
        anyhow::bail!("Swift executable did not report any dynamic dependencies");
    }
    let unsupported = dependencies
        .into_iter()
        .filter(|dependency| {
            !dependency.starts_with("/System/Library/")
                && !dependency.starts_with("/usr/lib/")
                && !dependency.starts_with("@rpath/libswift")
        })
        .collect::<Vec<_>>();
    if !unsupported.is_empty() {
        anyhow::bail!(
            "Swift executable has non-system dynamic dependencies: {}",
            unsupported.join(", ")
        );
    }
    Ok(())
}

fn archive_distribution(
    sh: &Shell,
    ctx: &BuildContext,
    xcframework: &Path,
) -> Result<(PathBuf, PathBuf)> {
    let archive = ctx.swift_artifacts_dir().join(SWIFT_DISTRIBUTION_ARCHIVE);
    output::run_build_command(
        "Archiving Swift XCFramework",
        cmd!(
            sh,
            "{DITTO_PATH} -c -k --sequesterRsrc --keepParent {xcframework} {archive}"
        ),
    )
    .context("failed to archive Swift XCFramework")?;

    let swift_checksum = cmd!(sh, "{XCRUN_PATH} swift package compute-checksum {archive}")
        .read()
        .context("SwiftPM failed to compute the XCFramework checksum")?;
    let swift_checksum = swift_checksum.trim();
    let actual_checksum = sha256_file(&archive)?;
    if swift_checksum != actual_checksum {
        anyhow::bail!("SwiftPM checksum {swift_checksum} did not match SHA-256 {actual_checksum}");
    }

    let checksum = ctx.swift_artifacts_dir().join(SWIFT_DISTRIBUTION_CHECKSUM);
    fs::write(&checksum, format!("{actual_checksum}\n"))
        .with_context(|| format!("failed to write checksum {}", checksum.display()))?;
    Ok((archive, checksum))
}
