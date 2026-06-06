use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use cogentlm_gateway::GatewayFileConfig;
use cogentlm_gateway_server::http;

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
        .with_context(|| format!("failed to load gateway config {}", config.display()))?;
    let bind = config.server.bind;
    let access = config
        .gateway_access()
        .context("failed to build gateway token access")?;
    let token = load_secret_env(&config.auth.token_env)?;
    let admin_env = config
        .auth
        .admin_token_env
        .clone()
        .context("auth.admin_token_env is required for the gateway dashboard")?;
    let admin_token = load_secret_env(&admin_env)?;
    let max_request_bytes = config
        .limits
        .max_request_bytes()
        .context("invalid gateway request byte limit")?;
    let history_capacity = config
        .limits
        .history_capacity()
        .context("invalid gateway history capacity")?;
    let allowed_origins = config.cors.allowed_origins.clone();
    let adapter = config
        .build_adapter()
        .await
        .context("failed to build gateway adapter")?;
    let service = http::GatewayHttpService::new(
        adapter,
        vec![http::GatewayToken::new(token, access)?],
        admin_token,
        allowed_origins,
        http::GatewayHttpLimits { max_request_bytes },
        history_capacity,
    )
    .context("failed to build gateway HTTP service")?;

    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind gateway to {bind}"))?;
    axum::serve(listener, service.router().into_make_service())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await
        .context("gateway server stopped with an error")
}

fn load_secret_env(name: &str) -> anyhow::Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} is required"))?;
    if value.trim().is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    Ok(value)
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
