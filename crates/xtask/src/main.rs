use anyhow::Result;
use clap::Parser;
use xshell::Shell;
use xtask::cli::{BuildCommands, Cli, Commands};
use xtask::targets;
use xtask::utils::BuildContext;

fn main() -> Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;
    let ctx = BuildContext::new()?;

    match cli.command {
        Commands::Build { target } => run_build(target, &sh, &ctx)?,
    }

    Ok(())
}

fn run_build(target: BuildCommands, sh: &Shell, ctx: &BuildContext) -> Result<()> {
    match target {
        BuildCommands::All => {
            targets::core::build(sh, ctx)?;
            targets::wasm::build(sh, ctx)?;
            targets::python::build(sh, ctx, None)?;
            targets::node::build(sh, ctx, None)?;
        }
        BuildCommands::Core => targets::core::build(sh, ctx)?,
        BuildCommands::Wasm => targets::wasm::build(sh, ctx)?,
        BuildCommands::Python(args) => targets::python::build(sh, ctx, args.backend.as_ref())?,
        BuildCommands::Node(args) => targets::node::build(sh, ctx, args.backend.as_ref())?,
    }

    Ok(())
}
