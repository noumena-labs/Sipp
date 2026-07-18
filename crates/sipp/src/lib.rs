//! Public Rust library for Sipp inference.
//!
//! This crate is the intended single dependency for Rust applications. It
//! exposes the high-level client API together with the native engine config
//! and shared value types needed to run local, gateway, or direct
//! provider inference.

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/root_tests.rs"]
mod root_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
mod chat;
mod client;
mod collection;
mod defaults;
mod native_bridge;

/// High-level client API for local, gateway, and provider endpoints.
pub use self::client::*;

/// Native backend helpers.
pub mod backend;

/// Provider-neutral value types shared across Sipp APIs.
pub mod core;

/// Native engine configuration and lower-level runtime APIs.
pub mod engine;

/// Engine and endpoint error types.
pub mod error;

#[cfg(feature = "gateway")]
/// Protocol-neutral gateway execution primitives.
pub mod gateway_core;

/// Native model lifecycle helpers and backend selection types.
pub mod lifecycle;

#[cfg(feature = "providers")]
/// Direct provider adapters behind the `providers` feature.
pub mod providers;

/// Native runtime execution APIs and request/metrics types.
pub mod runtime;

/// GGUF inspection and browser cache sharding utilities.
pub mod shard;

/// Common native runtime configuration and default generation constants.
pub use self::engine::{NativeRuntimeConfig, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS};

/// Returns the crate package version.
pub fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
