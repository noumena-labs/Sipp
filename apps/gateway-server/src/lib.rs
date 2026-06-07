//! Standalone CogentLM gateway server components.

/// Application-owned configuration and endpoint construction.
pub mod config;
/// Axum HTTP service for the standalone gateway server.
pub mod http;
/// Process lifecycle state shared by public and management listeners.
pub mod lifecycle;
/// Low-cardinality Prometheus metrics.
pub mod metrics;

#[cfg(test)]
#[path = "tests/config_tests.rs"]
mod config_tests;
#[cfg(test)]
#[path = "tests/deployment_tests.rs"]
mod deployment_tests;
#[cfg(test)]
#[path = "tests/http_tests.rs"]
mod http_tests;
