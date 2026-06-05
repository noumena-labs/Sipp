use std::path::PathBuf;

use anyhow::Context;
use axum::{routing::get, Router};
use clap::Parser;
use cogentlm_gateway::GatewayFileConfig;

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
        .with_context(|| format!("failed to load gateway config {}", cli.config.display()))?
        .build()
        .await
        .context("failed to build gateway service")?;
    let bind = gateway.bind;

    let app = Router::new()
        .route("/healthz", get(healthz))
        .merge(gateway.service.router());
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

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
}
