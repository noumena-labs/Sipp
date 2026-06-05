//! Shared helpers for the Rust example binaries.

/// Shared local-model example helpers.
pub mod local_common;

/// Shared remote gateway example helpers.
#[cfg(feature = "remote")]
pub mod remote_common;
