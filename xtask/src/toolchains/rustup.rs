//! Rustup-managed compilation targets required by Apple package builds.

use crate::output;
use anyhow::{Context, Result};
use xshell::{cmd, Shell};

pub(crate) const APPLE_TARGETS: [&str; 5] = [
    "aarch64-apple-darwin",
    "x86_64-apple-darwin",
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
    "x86_64-apple-ios",
];

/// Installs Rust standard libraries required by macOS and iOS builds.
pub(crate) fn setup_apple_targets(sh: &Shell) -> Result<()> {
    if !cfg!(target_os = "macos") {
        output::detail("Rust Apple targets", "skipped on non-macOS host");
        return Ok(());
    }

    output::run_build_command(
        "Installing Rust Apple targets",
        cmd!(sh, "rustup target add {APPLE_TARGETS...}"),
    )
    .context("failed to install Rust Apple targets")
}
