//! Native Rust library build target.

use crate::output;
use crate::toolchains::env::apply_toolchains;
use crate::utils::BuildContext;
use anyhow::Result;
use xshell::{cmd, Shell};

/// Builds the native Rust library crates.
pub fn build(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("Native Rust libraries");
    output::path("Workspace", ctx.workspace_root());
    output::path("Cargo target dir", &ctx.cargo_build_root());

    let _dir = sh.push_dir(ctx.workspace_root());
    let cargo_cmd = apply_toolchains(
        sh,
        ctx,
        cmd!(
            sh,
            "cargo build --release --package sipp-sys --package sipp-rs --package sipp-gateway --package sipp-binding-dto"
        ),
        None,
    )?;
    output::run_build_command("Building release Rust library crates", cargo_cmd)?;

    Ok(())
}
