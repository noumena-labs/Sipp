use anyhow::Result;
use clap::Parser;
use xshell::Shell;
use xtask::cli::{Backend, BuildCommands, Cli, Commands, DocsCommands, TestCommands};
use xtask::targets;
use xtask::utils::BuildContext;
use xtask::{clean, configure_output, docs, doctor, finish_output, run, setup, test, toolchain};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/main_tests.rs"]
mod main_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

fn main() -> Result<()> {
    let cli = Cli::parse();
    let sh = Shell::new()?;
    let ctx = BuildContext::new()?;
    let output = effective_output_options(&cli.command, cli.verbose, cli.plain);
    configure_output(&ctx, output.stream_subprocess, cli.no_banner, output.plain);
    let summary = command_summary(&cli.command);

    let result = match cli.command {
        Commands::Build { target } => run_build(target, &sh, &ctx),
        Commands::Clean(args) => clean::run(&sh, &ctx, &args),
        Commands::Run { command } => run::run(&sh, &ctx, command),
        Commands::Test { command } => test::run(&sh, &ctx, command),
        Commands::Toolchain { command } => toolchain::run(&sh, &ctx, command),
        Commands::Doctor(args) => doctor::run(&ctx, &args),
        Commands::Setup(args) => setup::run(&sh, &ctx, &args),
        Commands::Docs { command } => run_docs(command, &sh, &ctx),
    };
    if output.final_status {
        finish_output(result.is_ok(), &summary);
    }

    result
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct OutputOptions {
    stream_subprocess: bool,
    plain: bool,
    final_status: bool,
}

fn effective_output_options(command: &Commands, verbose: bool, plain: bool) -> OutputOptions {
    let compact_default = matches!(
        command,
        Commands::Build { .. }
            | Commands::Run { .. }
            | Commands::Docs { .. }
            | Commands::Test {
                command: TestCommands::Unit(_) | TestCommands::Smoke(_),
            }
    );
    let machine_readable = matches!(
        command,
        Commands::Test {
            command: TestCommands::List(args),
        } if args.format == xtask::cli::TestListFormat::Json
    );
    OutputOptions {
        stream_subprocess: verbose,
        plain: plain || !compact_default || machine_readable,
        final_status: !machine_readable,
    }
}

fn command_summary(command: &Commands) -> String {
    match command {
        Commands::Build { target } => build_summary(target),
        Commands::Clean(_) => "Clean workspace".to_owned(),
        Commands::Run { .. } => "Run developer workflow".to_owned(),
        Commands::Test { command } => test_summary(command),
        Commands::Toolchain { .. } => "Toolchain command".to_owned(),
        Commands::Doctor(_) => "Developer environment doctor".to_owned(),
        Commands::Setup(_) => "CogentLM setup".to_owned(),
        Commands::Docs { command } => docs_summary(command),
    }
}

fn docs_summary(command: &DocsCommands) -> String {
    match command {
        DocsCommands::Build => "Build documentation book".to_owned(),
        DocsCommands::Serve => "Build and serve documentation book".to_owned(),
    }
}

fn test_summary(command: &TestCommands) -> String {
    match command {
        TestCommands::List(_) => "List test suites".to_owned(),
        TestCommands::Unit(_) => "Run unit tests".to_owned(),
        TestCommands::Smoke(_) => "Run smoke tests".to_owned(),
        TestCommands::Verify(_) => "Verify tests".to_owned(),
    }
}

fn build_summary(target: &BuildCommands) -> String {
    match target {
        BuildCommands::All => "Build all default targets".to_owned(),
        BuildCommands::Core => "Build native Rust workspace".to_owned(),
        BuildCommands::Wasm => "Build browser WASM/WebGPU package".to_owned(),
        BuildCommands::Python(args) => {
            format!("Build Python bindings ({})", backend_summary(args.backend))
        }
        BuildCommands::Node(args) => {
            format!("Build Node.js bindings ({})", backend_summary(args.backend))
        }
        BuildCommands::Cli(args) => {
            format!(
                "Build Rust CLI distribution ({})",
                backend_summary(args.backend)
            )
        }
        BuildCommands::GatewayServer(args) => {
            format!("Build gateway-server ({})", backend_summary(args.backend))
        }
    }
}

fn backend_summary(backend: Option<Backend>) -> &'static str {
    backend.as_ref().map(Backend::as_str).unwrap_or("cpu")
}

fn run_docs(command: DocsCommands, sh: &Shell, ctx: &BuildContext) -> Result<()> {
    match command {
        DocsCommands::Build => docs::run_build(sh, ctx),
        DocsCommands::Serve => docs::run_serve(sh, ctx),
    }
}

fn run_build(target: BuildCommands, sh: &Shell, ctx: &BuildContext) -> Result<()> {
    match target {
        BuildCommands::All => {
            targets::core::build(sh, ctx)?;
            targets::wasm::build(sh, ctx)?;
            targets::python::build(sh, ctx, None)?;
            targets::node::build(sh, ctx, None)?;
            targets::cli::build(sh, ctx, None)?;
        }
        BuildCommands::Core => targets::core::build(sh, ctx)?,
        BuildCommands::Wasm => targets::wasm::build(sh, ctx)?,
        BuildCommands::Python(args) => targets::python::build(sh, ctx, args.backend.as_ref())?,
        BuildCommands::Node(args) => targets::node::build(sh, ctx, args.backend.as_ref())?,
        BuildCommands::Cli(args) => targets::cli::build(sh, ctx, args.backend.as_ref())?,
        BuildCommands::GatewayServer(args) => {
            targets::gateway_server::build(sh, ctx, args.backend.as_ref())?
        }
    }

    Ok(())
}
