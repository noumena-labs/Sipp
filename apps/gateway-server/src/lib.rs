//! First-party CogentLM gateway application components.

/// Application-owned configuration and endpoint construction.
pub mod config;
/// Axum HTTP service for the standalone gateway server.
pub mod http;
/// Application-owned metrics.
pub mod metrics;

#[cfg(test)]
#[path = "tests/server_tests.rs"]
mod tests;
