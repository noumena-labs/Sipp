//! Python binding build target.

use crate::cli::Backend;
use crate::toolchains::env::apply_toolchains;
use crate::utils::BuildContext;
use anyhow::Result;
use xshell::{cmd, Shell};

/// Builds the Python bindings for the selected backend.
pub fn build(sh: &Shell, ctx: &BuildContext, backend: Option<&Backend>) -> Result<()> {
    if matches!(backend, Some(Backend::All)) {
        anyhow::bail!("--backend all is only supported for Node bindings");
    }

    println!("=> Building Python Bindings...");
    let _dir = sh.push_dir(ctx.workspace_root().join("bindings").join("python"));
    let target_dir = ctx.cargo_python_target_dir(backend);
    sh.create_dir(&target_dir)?;

    let mut maturin_cmd =
        cmd!(sh, "uv run maturin develop --release").env("CARGO_TARGET_DIR", &target_dir);
    maturin_cmd = apply_toolchains(sh, ctx, maturin_cmd, backend)?;

    match backend {
        Some(Backend::Cpu) | None => {
            println!("   Hardware Backend: CPU (Default)");
        }
        Some(b) => {
            let feature = b.as_str();
            println!("   Hardware Backend: {}", feature.to_uppercase());
            maturin_cmd = maturin_cmd.arg("--features").arg(feature);
        }
    }

    maturin_cmd.run()?;
    Ok(())
}
