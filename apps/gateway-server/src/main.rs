use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, Subcommand};
use cogentlm_gateway_server::{
    config::GatewayServerConfig, http::GatewayHttpService, metrics::GatewayMetrics,
};
use tokio::sync::oneshot;

#[derive(Debug, Parser)]
#[command(name = "cogentlm-gateway")]
#[command(about = "Run the first-party CogentLM gateway application")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the gateway application.
    Serve {
        /// Path to the application TOML file.
        #[arg(long)]
        config: PathBuf,
    },
    /// Parse and validate configuration without loading endpoints.
    Check {
        /// Path to the application TOML file.
        #[arg(long)]
        config: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();
    match Cli::parse().command {
        Command::Serve { config } => serve(config).await,
        Command::Check { config } => check(config),
    }
}

fn check(path: PathBuf) -> anyhow::Result<()> {
    GatewayServerConfig::from_path(&path)?;
    println!("configuration is valid: {}", path.display());
    Ok(())
}

async fn serve(path: PathBuf) -> anyhow::Result<()> {
    let config = GatewayServerConfig::from_path(&path)?;
    let runtime = config.build_runtime().await?;
    let tokens = config.load_tokens()?;
    let admin_password = config.load_admin_password()?;
    let metrics = Arc::new(GatewayMetrics::new());
    let service = GatewayHttpService::new(
        runtime,
        config.routes.clone(),
        tokens,
        admin_password,
        metrics,
        config.max_request_bytes,
        &config.allowed_origins,
        config.max_concurrent_requests,
        config.security.clone(),
        admin_assets_dir()?,
    )?;

    let management_listener = tokio::net::TcpListener::bind(config.management_bind)
        .await
        .with_context(|| {
            format!(
                "failed to bind management listener {}",
                config.management_bind
            )
        })?;
    let public_listener = tokio::net::TcpListener::bind(config.public_bind)
        .await
        .with_context(|| format!("failed to bind public listener {}", config.public_bind))?;

    let (management_stop_tx, management_stop_rx) = oneshot::channel();
    let management_router = service.management_router();
    let management_task = tokio::spawn(async move {
        axum::serve(management_listener, management_router.into_make_service())
            .with_graceful_shutdown(async {
                let _ = management_stop_rx.await;
            })
            .await
    });

    let public_router = service.public_router();
    let public_task = tokio::spawn(async move {
        axum::serve(
            public_listener,
            public_router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
        )
        .with_graceful_shutdown(shutdown_signal())
        .await
    });

    tracing::info!(
        public_bind = %config.public_bind,
        management_bind = %config.management_bind,
        "gateway application ready"
    );
    public_task.await??;
    let _ = management_stop_tx.send(());
    management_task.await??;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    if let Err(error) = tracing_subscriber::fmt()
        .json()
        .with_env_filter(filter)
        .try_init()
    {
        eprintln!("failed to initialize gateway tracing: {error}");
    }
}

fn admin_assets_dir() -> anyhow::Result<PathBuf> {
    if let Some(path) = std::env::var_os("COGENTLM_GATEWAY_ADMIN_ASSETS_DIR") {
        return Ok(PathBuf::from(path));
    }

    let executable = std::env::current_exe()?;
    let executable_dir = executable.parent().with_context(|| {
        format!(
            "failed to resolve parent directory of {}",
            executable.display()
        )
    })?;
    Ok(executable_dir.join("admin-ui"))
}
