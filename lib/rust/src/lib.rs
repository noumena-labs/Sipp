//! Public Rust facade for CogentLM inference.
//!
//! This crate is the intended single dependency for Rust applications. It
//! re-exports the high-level client API together with the native engine config
//! and shared value types needed to run local or gateway-backed inference.

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/root_tests.rs"]
mod root_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
/// High-level client API for local models and remote gateway endpoints.
pub use cogentlm_client::*;

/// Native backend helpers.
pub mod backend {
    pub use cogentlm_engine::backend::*;
}

/// Provider-neutral value types shared across CogentLM APIs.
pub mod core {
    pub use cogentlm_core::*;
}

/// Native engine configuration and lower-level runtime APIs.
pub mod engine {
    pub use cogentlm_engine::engine::*;
}

/// Native model lifecycle helpers and backend selection types.
pub mod lifecycle {
    pub use cogentlm_engine::lifecycle::*;
}

/// Native runtime execution APIs and request/metrics types.
pub mod runtime {
    pub use cogentlm_engine::runtime::*;
}

/// GGUF inspection and browser cache sharding utilities.
pub mod shard {
    pub use cogentlm_shard::*;
}

/// Common native runtime configuration and default generation constants.
pub use cogentlm_engine::engine::{NativeRuntimeConfig, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS};

/// Returns the facade crate package version.
pub fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
