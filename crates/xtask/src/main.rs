use anyhow::Result;
use clap::Parser;
use xshell::Shell;
use xtask::cli::{Cli, Commands};
use xtask::targets;
use xtask::utils::BuildContext;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;
    let ctx = BuildContext::new()?;

    match cli.command {
        Commands::BuildAll => {
            targets::core::build(&sh, &ctx)?;
            targets::wasm::build(&sh, &ctx)?;
            targets::python::build(&sh, &ctx, None)?;
            targets::node::build(&sh, &ctx, None)?;
        }
        Commands::BuildCore => targets::core::build(&sh, &ctx)?,
        Commands::BuildWasm => targets::wasm::build(&sh, &ctx)?,
        Commands::BuildPython { backend } => targets::python::build(&sh, &ctx, backend.as_ref())?,
        Commands::BuildNode { backend } => targets::node::build(&sh, &ctx, backend.as_ref())?,
    }

    Ok(())
}
