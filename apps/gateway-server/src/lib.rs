//! First-party Sipp gateway application components.

pub(crate) mod admin;
/// Application-owned configuration and endpoint construction.
pub mod config;
/// Axum HTTP service for the standalone gateway server.
pub mod http;
/// Application-owned metrics.
pub mod metrics;
pub(crate) mod runtime;

#[cfg(test)]
#[path = "tests/server_tests.rs"]
mod tests;
