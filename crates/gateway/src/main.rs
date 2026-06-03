use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use cogentlm_gateway::GatewayFileConfig;

#[derive(Debug, Parser)]
#[command(name = "cogentlm-gateway")]
#[command(about = "Run a CogentLM remote gateway")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the gateway using a TOML configuration file.
    Serve {
        /// Path to gateway.toml.
        #[arg(long)]
        config: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Serve { config } => serve(config).await,
    }
}

async fn serve(config: PathBuf) -> anyhow::Result<()> {
    let config = GatewayFileConfig::from_path(&config)
        .with_context(|| format!("failed to load gateway config {}", config.display()))?
        .build()
        .await
        .context("failed to build gateway service")?;
    let listener = tokio::net::TcpListener::bind(config.bind)
        .await
        .with_context(|| format!("failed to bind gateway to {}", config.bind))?;
    axum::serve(listener, config.service.router()?.into_make_service())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("gateway server stopped with an error")
}

fn init_tracing() {
    let filter = match tracing_subscriber::EnvFilter::try_from_default_env() {
        Ok(filter) => filter,
        Err(_) => tracing_subscriber::EnvFilter::new("info"),
    };
    if let Err(error) = tracing_subscriber::fmt().with_env_filter(filter).try_init() {
        eprintln!("failed to initialize gateway tracing: {error}");
    }
}
