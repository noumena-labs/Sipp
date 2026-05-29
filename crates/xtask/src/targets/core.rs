//! Native Rust workspace build target.

use crate::output;
use crate::utils::BuildContext;
use anyhow::Result;
use xshell::{cmd, Shell};

/// Builds the native Rust workspace crates.
pub fn build(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("Native Rust workspace");
    output::path("Workspace", ctx.workspace_root());
    output::path("Cargo target dir", &ctx.cargo_build_root());

    let _dir = sh.push_dir(ctx.workspace_root());
    output::run_command(
        "Building release workspace crates",
        cmd!(sh, "cargo build --release --workspace --exclude xtask"),
    )?;

    Ok(())
}
