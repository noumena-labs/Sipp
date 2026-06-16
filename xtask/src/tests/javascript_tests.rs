//! Tests the `javascript` module in `xtask`.
//!
//! Covers JavaScript workspace dependency helpers for package manifest parsing,
//! filtered root workspace installs, and deterministic duplicate handling.

use crate::test_support::TempDir;

use super::{package_name, root_workspace_package_filters};

#[test]
fn package_name_reads_manifest_name() {
    let temp = TempDir::new("javascript-package-name");
    let package_dir = temp.create_dir("lib/web");
    temp.write(
        "lib/web/package.json",
        r#"{"name":"@noumena-labs/sipp","version":"0.1.0"}"#,
    );

    assert_eq!(package_name(&package_dir).unwrap(), "@noumena-labs/sipp");
}

#[test]
fn package_name_requires_nonempty_name() {
    let temp = TempDir::new("javascript-package-name-missing");
    let package_dir = temp.create_dir("lib/web");
    temp.write("lib/web/package.json", r#"{"version":"0.1.0"}"#);

    let error = package_name(&package_dir).unwrap_err();
    assert!(format!("{error:#}").contains("missing package name"));
}

#[test]
fn root_workspace_package_filters_preserve_order_and_remove_duplicates() {
    let temp = TempDir::new("javascript-filters");
    let web_dir = temp.create_dir("lib/web");
    let demo_dir = temp.create_dir("demos/chat");
    temp.write("lib/web/package.json", r#"{"name":"@noumena-labs/sipp"}"#);
    temp.write("demos/chat/package.json", r#"{"name":"sipp-chat-demo"}"#);

    let filters = root_workspace_package_filters(&[demo_dir.clone(), web_dir, demo_dir]).unwrap();

    assert_eq!(
        filters,
        vec!["sipp-chat-demo".to_owned(), "@noumena-labs/sipp".to_owned()]
    );
}

#[test]
fn root_workspace_package_filters_reject_empty_package_set() {
    let error = root_workspace_package_filters(&[]).unwrap_err();

    assert!(format!("{error:#}").contains("at least one JavaScript workspace"));
}
