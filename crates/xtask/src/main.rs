use anyhow::Result;
use clap::Parser;
use xshell::Shell;
use xtask::cli::{BuildCommands, Cli, Commands};
use xtask::targets;
use xtask::utils::BuildContext;
use xtask::{clean, configure_output, doctor, finish_output, run, setup, toolchain};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;
    let ctx = BuildContext::new()?;
    configure_output(&ctx, cli.verbose, cli.no_banner, cli.plain);

    let result = match cli.command {
        Commands::Build { target } => run_build(target, &sh, &ctx),
        Commands::Clean(args) => clean::run(&sh, &ctx, &args),
        Commands::Run { command } => run::run(&sh, &ctx, command),
        Commands::Toolchain { command } => toolchain::run(&sh, &ctx, command),
        Commands::Doctor(args) => doctor::run(&ctx, &args),
        Commands::Setup(args) => setup::run(&sh, &ctx, &args),
    };
    finish_output();

    result
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
