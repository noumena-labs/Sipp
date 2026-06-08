//! Tests the `targets::gateway_server` module in `xtask`.
//!
//! Covers gateway artifact caching helpers with fixture directories instead of
//! running Bun or compiling the gateway binary.

use crate::cli::Backend;
use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{
    admin_ui_fingerprint, backend_label, backends_to_build, clean_stale_runtime_artifacts,
    copy_admin_ui, files_equal, gateway_binary_file_name, is_gateway_runtime_artifact,
    prepare_dist_dir, replace_file,
};

#[test]
fn backend_expansion_and_labels_are_stable() {
    assert_eq!(backends_to_build(None), vec![Backend::Cpu]);
    assert_eq!(backends_to_build(Some(&Backend::Cuda)), vec![Backend::Cuda]);
    if cfg!(target_os = "macos") {
        assert_eq!(
            backends_to_build(Some(&Backend::All)),
            vec![Backend::Cpu, Backend::Metal]
        );
    } else {
        assert_eq!(
            backends_to_build(Some(&Backend::All)),
            vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda]
        );
    }
    assert_eq!(backend_label(None), "cpu (default)");
    assert_eq!(backend_label(Some(&Backend::Vulkan)), "vulkan");
}

#[test]
fn prepare_dist_dir_preserves_existing_artifacts() {
    let temp = TempDir::new("target-gateway-prepare");
    let sh = xshell::Shell::new().unwrap();
    let dist = temp.create_dir("dist");
    temp.write("dist/keep.txt", "keep");

    prepare_dist_dir(&sh, &dist).unwrap();

    assert_eq!(
        std::fs::read_to_string(dist.join("keep.txt")).unwrap(),
        "keep"
    );
}

#[test]
fn replace_file_skips_equal_content_and_updates_changed_content() {
    let temp = TempDir::new("target-gateway-replace");
    let sh = xshell::Shell::new().unwrap();
    let source = temp.write("source.txt", "same");
    let dest = temp.write("dest.txt", "same");

    assert!(files_equal(&source, &dest).unwrap());
    replace_file(&sh, &source, &dest).unwrap();
    assert_eq!(std::fs::read_to_string(&dest).unwrap(), "same");

    temp.write("source.txt", "changed");
    replace_file(&sh, &source, &dest).unwrap();
    assert_eq!(std::fs::read_to_string(&dest).unwrap(), "changed");
}

#[test]
fn copy_admin_ui_copies_changed_files_and_removes_stale_files() {
    let temp = TempDir::new("target-gateway-admin-copy");
    let sh = xshell::Shell::new().unwrap();
    let source = temp.create_dir("source");
    let dest = temp.create_dir("dest");
    temp.write("source/index.html", "index");
    temp.write("source/assets/app.js", "app");
    temp.write("dest/stale.js", "stale");
    temp.write("dest/assets/app.js", "old");

    copy_admin_ui(&sh, &source, &dest).unwrap();

    assert_eq!(
        std::fs::read_to_string(dest.join("index.html")).unwrap(),
        "index"
    );
    assert_eq!(
        std::fs::read_to_string(dest.join("assets/app.js")).unwrap(),
        "app"
    );
    assert!(!dest.join("stale.js").exists());
}

#[test]
fn admin_ui_fingerprint_ignores_dist_and_node_modules() {
    let temp = TempDir::new("target-gateway-admin-fingerprint");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let admin = temp.create_dir("apps/gateway-server/admin-ui");
    temp.write("package.json", "{}");
    temp.write("bun.lock", "lock");
    temp.write("apps/gateway-server/admin-ui/src/main.ts", "main");
    temp.write("apps/gateway-server/admin-ui/dist/index.html", "old");
    temp.write(
        "apps/gateway-server/admin-ui/node_modules/pkg/index.js",
        "old",
    );

    let first = admin_ui_fingerprint(&ctx, &admin).unwrap();
    temp.write("apps/gateway-server/admin-ui/dist/index.html", "changed");
    temp.write(
        "apps/gateway-server/admin-ui/node_modules/pkg/index.js",
        "changed",
    );
    let second = admin_ui_fingerprint(&ctx, &admin).unwrap();
    temp.write("apps/gateway-server/admin-ui/src/main.ts", "changed");
    let third = admin_ui_fingerprint(&ctx, &admin).unwrap();

    assert_eq!(first, second);
    assert_ne!(second, third);
}

#[test]
fn stale_runtime_cleanup_removes_only_recognized_unexpected_files() {
    let temp = TempDir::new("target-gateway-stale-runtime");
    let sh = xshell::Shell::new().unwrap();
    let dist = temp.create_dir("dist");
    let binary = gateway_binary_file_name();
    let plugin = if cfg!(windows) {
        "ggml-vulkan.dll"
    } else {
        "libggml-vulkan.so"
    };
    let stale_plugin = if cfg!(windows) {
        "ggml-cuda.dll"
    } else {
        "libggml-cuda.so"
    };
    temp.write(format!("dist/{binary}"), "bin");
    temp.write(format!("dist/{plugin}"), "vulkan");
    temp.write(format!("dist/{stale_plugin}"), "cuda");
    temp.write("dist/README.txt", "keep");

    let expected = [binary.to_string(), plugin.to_string()]
        .into_iter()
        .collect();
    clean_stale_runtime_artifacts(&sh, &dist, &expected).unwrap();

    assert!(dist.join(binary).exists());
    assert!(dist.join(plugin).exists());
    assert!(!dist.join(stale_plugin).exists());
    assert!(dist.join("README.txt").exists());
    assert!(is_gateway_runtime_artifact(plugin));
    assert!(!is_gateway_runtime_artifact("README.txt"));
}
