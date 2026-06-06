use std::path::PathBuf;

use anyhow::Context;
use axum::{routing::get, Router};
use clap::Parser;
use cogentlm_gateway::GatewayFileConfig;
use cogentlm_gateway_server::http::{GatewayHttpLimits, GatewayHttpService, GatewayToken};

#[derive(Debug, Parser)]
#[command(name = "cogentlm-gateway-example")]
#[command(about = "Run a CogentLM gateway inside an application server")]
struct Cli {
    /// Path to a gateway TOML configuration.
    #[arg(long)]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let gateway = GatewayFileConfig::from_path(&cli.config)
        .with_context(|| format!("failed to load gateway config {}", cli.config.display()))?;
    let bind = gateway.server.bind;
    let access = gateway
        .gateway_access()
        .context("failed to build gateway token access")?;
    let token = required_env(&gateway.auth.token_env)?;
    let admin_env = gateway
        .auth
        .admin_token_env
        .as_deref()
        .context("auth.admin_token_env is required")?;
    let admin_token = required_env(admin_env)?;
    let max_request_bytes = gateway.limits.max_request_bytes()?;
    let history_capacity = gateway.limits.history_capacity()?;
    let allowed_origins = gateway.cors.allowed_origins.clone();
    let adapter = gateway
        .build_adapter()
        .await
        .context("failed to build gateway adapter")?;
    let gateway = GatewayHttpService::new(
        adapter,
        vec![GatewayToken::new(token, access)?],
        admin_token,
        allowed_origins,
        GatewayHttpLimits { max_request_bytes },
        history_capacity,
    )
    .context("failed to build gateway HTTP service")?;

    let app = Router::new()
        .route("/app-healthz", get(healthz))
        .merge(gateway.router());
    let listener = tokio::net::TcpListener::bind(bind)
        .await
        .with_context(|| format!("failed to bind gateway example to {bind}"))?;

    println!("gateway example listening on {bind}");
    axum::serve(listener, app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("gateway example server stopped with an error")
}

async fn healthz() -> &'static str {
    "ok"
}

fn required_env(name: &str) -> anyhow::Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} is required"))?;
    if value.trim().is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    Ok(value)
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
