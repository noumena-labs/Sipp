//! Tests the `toolchains::env` module in `xtask`.
//!
//! Covers deterministic platform path-separator selection without constructing
//! commands that would bootstrap or inspect external toolchains.

use super::{macos_deployment_target, path_separator};

#[test]
fn path_separator_matches_host_platform() {
    assert_eq!(path_separator(), if cfg!(windows) { ";" } else { ":" });
}

#[test]
fn macos_deployment_target_matches_host_architecture() {
    let expected = if cfg!(target_os = "macos") {
        if cfg!(target_arch = "aarch64") {
            Some("11.0")
        } else {
            Some("10.15")
        }
    } else {
        None
    };

    assert_eq!(macos_deployment_target(), expected);
}
