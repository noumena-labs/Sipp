use std::path::PathBuf;

use anyhow::Context;
use clap::{Parser, Subcommand};
use cogentlm_gateway_server::{config::GatewayServerConfig, http::GatewayHttpService};
use tokio::sync::oneshot;

#[derive(Debug, Parser)]
#[command(name = "cogentlm-gateway")]
#[command(about = "Run a production CogentLM gateway service")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Serve the gateway.
    Serve {
        /// Path to the gateway TOML file.
        #[arg(long)]
        config: PathBuf,
    },
    /// Parse and validate configuration without reading secrets or loading endpoints.
    Check {
        /// Path to the gateway TOML file.
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
    let tokens = config.load_tokens()?;
    let service =
        GatewayHttpService::new(tokens, config.max_request_bytes, &config.allowed_origins)?;

    let (management_stop_tx, management_stop_rx) = oneshot::channel();
    let management_router = service.management_router();
    let management_task = tokio::spawn(async move {
        axum::serve(management_listener, management_router.into_make_service())
            .with_graceful_shutdown(async {
                let _ = management_stop_rx.await;
            })
            .await
    });

    let (public_stop_tx, public_stop_rx) = oneshot::channel();
    let public_router = service.public_router();
    let mut public_task = tokio::spawn(async move {
        axum::serve(public_listener, public_router.into_make_service())
            .with_graceful_shutdown(async {
                let _ = public_stop_rx.await;
            })
            .await
    });

    tracing::info!(
        public_bind = %config.public_bind,
        management_bind = %config.management_bind,
        "gateway listeners bound; loading configured endpoints"
    );

    match config.build_adapter().await {
        Ok(adapter) => service.set_ready(adapter).await,
        Err(error) => {
            service.set_failed();
            tracing::error!(error = %error, "gateway endpoint loading failed");
            let _ = public_stop_tx.send(());
            let _ = management_stop_tx.send(());
            let _ = public_task.await;
            let _ = management_task.await;
            return Err(error);
        }
    }
    tracing::info!("gateway is ready");

    shutdown_signal().await?;
    service.begin_draining();
    tracing::info!(
        drain_timeout_seconds = config.drain_timeout().as_secs(),
        force_close_timeout_seconds = config.force_close_timeout().as_secs(),
        "gateway draining started"
    );
    let _ = public_stop_tx.send(());

    if !service.wait_for_idle(config.drain_timeout()).await {
        tracing::warn!("gateway drain deadline reached; cancelling active inference");
        service.cancel_active_for_shutdown();
        if !service.wait_for_idle(config.force_close_timeout()).await {
            tracing::warn!("gateway force-close deadline reached");
            public_task.abort();
        }
    }

    let _ = management_stop_tx.send(());
    if !public_task.is_finished() {
        let force_timeout = config.force_close_timeout();
        if tokio::time::timeout(force_timeout, &mut public_task)
            .await
            .is_err()
        {
            public_task.abort();
        }
    }
    let _ = management_task.await;
    tracing::info!("gateway shutdown complete");
    Ok(())
}

async fn shutdown_signal() -> anyhow::Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let mut terminate =
            signal(SignalKind::terminate()).context("failed to install SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => {
                result.context("failed to listen for Ctrl-C")?;
            }
            _ = terminate.recv() => {}
        }
    }

    #[cfg(not(unix))]
    {
        tokio::signal::ctrl_c()
            .await
            .context("failed to listen for Ctrl-C")?;
    }

    Ok(())
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
