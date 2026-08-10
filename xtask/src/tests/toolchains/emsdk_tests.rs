//! Tests the `toolchains::emsdk` module in `xtask`.
//!
//! Covers Emscripten install marker parsing, active-state checks, managed
//! Python command construction, and Windows patch idempotence with fixture
//! files instead of cloning or installing emsdk.

use crate::test_support::TempDir;

use super::{
    emdawnwebgpu_is_installed, emdawnwebgpu_package_dir, emsdk_is_active, emsdk_is_installed,
    emsdk_python_command, expected_emsdk_tool_id, patch_emsdk_windows, rust_target_list_contains,
    EMDAWNWEBGPU_VERSION, EMSDK_VERSION,
};

#[test]
fn expected_tool_id_uses_release_hash_from_tags_json() {
    let temp = TempDir::new("emsdk-tool-id");
    temp.write(
        "emscripten-releases-tags.json",
        format!(r#"{{"releases":{{"{EMSDK_VERSION}":"abc123"}}}}"#),
    );

    assert_eq!(
        expected_emsdk_tool_id(temp.path()).unwrap(),
        "releases-abc123-64bit"
    );
}

#[test]
fn installed_and_active_checks_require_version_and_required_paths() {
    let temp = TempDir::new("emsdk-installed");
    temp.write(
        "emscripten-releases-tags.json",
        format!(r#"{{"releases":{{"{EMSDK_VERSION}":"hash"}}}}"#),
    );
    temp.write("upstream/.emsdk_version", "releases-hash-64bit\n");
    temp.write("upstream/emscripten/emcc.bat", "");
    temp.write("upstream/emscripten/emcmake.bat", "");
    temp.write("upstream/emscripten/emmake.bat", "");
    temp.create_dir("node");
    temp.create_dir("python");

    assert!(emsdk_is_installed(temp.path()).unwrap());
    assert!(!emsdk_is_active(temp.path()).unwrap());
    temp.write(".emscripten", "");
    assert!(emsdk_is_active(temp.path()).unwrap());
}

#[test]
fn installed_check_rejects_wrong_version_marker() {
    let temp = TempDir::new("emsdk-wrong-version");
    temp.write(
        "emscripten-releases-tags.json",
        format!(r#"{{"releases":{{"{EMSDK_VERSION}":"hash"}}}}"#),
    );
    temp.write("upstream/.emsdk_version", "other\n");

    assert!(!emsdk_is_installed(temp.path()).unwrap());
}

#[test]
fn emdawnwebgpu_install_check_requires_port_and_version_marker() {
    let temp = TempDir::new("emdawnwebgpu-installed");
    let ctx = crate::utils::BuildContext::from_workspace_root_for_test(temp.path());
    let package_dir = emdawnwebgpu_package_dir(&ctx);

    assert_eq!(
        package_dir,
        temp.join(format!(
            ".build/toolchain/emdawnwebgpu/{EMDAWNWEBGPU_VERSION}/emdawnwebgpu_pkg"
        ))
    );
    assert!(!emdawnwebgpu_is_installed(&package_dir).unwrap());

    temp.write(
        format!(
            ".build/toolchain/emdawnwebgpu/{EMDAWNWEBGPU_VERSION}/emdawnwebgpu_pkg/emdawnwebgpu.port.py"
        ),
        "",
    );
    temp.write(
        format!(
            ".build/toolchain/emdawnwebgpu/{EMDAWNWEBGPU_VERSION}/emdawnwebgpu_pkg/VERSION.txt"
        ),
        format!("Dawn release {EMDAWNWEBGPU_VERSION} at revision abc."),
    );

    assert!(emdawnwebgpu_is_installed(&package_dir).unwrap());
}

#[test]
fn rust_target_list_matching_requires_exact_line() {
    let installed = "aarch64-apple-darwin\nwasm32-unknown-emscripten\n";

    assert!(rust_target_list_contains(
        installed,
        "wasm32-unknown-emscripten"
    ));
    assert!(!rust_target_list_contains(installed, "wasm32-unknown"));
}

#[test]
fn emsdk_python_command_uses_the_managed_interpreter_directly() {
    use std::ffi::{OsStr, OsString};

    let temp = TempDir::new("emsdk-managed-python");
    let python_exe = temp.join("managed-python/python.exe");
    let sh = xshell::Shell::new().unwrap();
    let command: std::process::Command =
        emsdk_python_command(&sh, &python_exe, temp.path(), "install").into();
    let args = command
        .get_args()
        .map(OsStr::to_os_string)
        .collect::<Vec<_>>();

    assert_eq!(command.get_program(), python_exe.as_os_str());
    assert_eq!(
        args,
        vec![
            temp.join("emsdk.py").into_os_string(),
            OsString::from("install"),
            OsString::from(EMSDK_VERSION),
        ]
    );
    assert!(command
        .get_envs()
        .any(|(key, value)| key == "PYTHONHOME" && value.is_none()));
    assert!(command
        .get_envs()
        .any(|(key, value)| key == "PYTHONPATH" && value.is_none()));
}

#[test]
fn windows_platform_patch_rewrites_expected_source_block_once() {
    let temp = TempDir::new("emsdk-patch");
    let old = "\
# platform.machine() may return AMD64 on windows, so standardize the case.
machine = os.getenv('EMSDK_ARCH', platform.machine().lower())
";
    temp.write("emsdk.py", old);

    patch_emsdk_windows(temp.path()).unwrap();
    let patched = std::fs::read_to_string(temp.join("emsdk.py")).unwrap();
    assert!(patched.contains("machine = os.getenv('EMSDK_ARCH')"));
    assert!(patched.contains("if not machine:"));

    patch_emsdk_windows(temp.path()).unwrap();
    assert_eq!(
        std::fs::read_to_string(temp.join("emsdk.py")).unwrap(),
        patched
    );
}
