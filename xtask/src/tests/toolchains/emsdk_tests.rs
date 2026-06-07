//! Tests the `toolchains::emsdk` module in `xtask`.
//!
//! Covers Emscripten install marker parsing, active-state checks, and Windows
//! patch idempotence with fixture files instead of cloning or installing emsdk.

use crate::test_support::TempDir;

use super::{
    emsdk_is_active, emsdk_is_installed, expected_emsdk_tool_id, patch_emsdk_windows, EMSDK_VERSION,
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
