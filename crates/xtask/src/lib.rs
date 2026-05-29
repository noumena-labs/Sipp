//! Build orchestration support for the CogentLM workspace.
//!
//! The binary entrypoint parses CLI commands and delegates to these modules so
//! target-specific build logic and host toolchain setup stay isolated.

pub mod cli;
pub mod targets;
pub mod toolchains;
pub mod utils;
