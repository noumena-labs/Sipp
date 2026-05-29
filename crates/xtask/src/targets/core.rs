//! Native Rust workspace build target.

use crate::utils::BuildContext;
use anyhow::Result;
use xshell::{cmd, Shell};

/// Builds the native Rust workspace crates.
pub fn build(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    println!("=> Building Native Rust Workspace...");
    let _dir = sh.push_dir(ctx.workspace_root());
    cmd!(sh, "cargo build --release --workspace --exclude xtask").run()?;
    Ok(())
}
