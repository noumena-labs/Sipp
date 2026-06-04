//! Build orchestration support for the CogentLM workspace.
//!
//! The binary entrypoint parses CLI commands and delegates to these modules so
//! target-specific build logic and host toolchain setup stay isolated.

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/support.rs"]
pub(crate) mod test_support;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////
pub mod clean;
pub mod cli;
pub mod doctor;
pub mod run;
pub mod sample_model;
pub mod setup;
pub mod targets;
pub(crate) mod terminal;
pub mod test;
pub mod toolchain;
pub mod toolchains;
pub mod utils;

pub(crate) use terminal as output;

/// Configures terminal output for the xtask binary entrypoint.
pub fn configure_output(ctx: &utils::BuildContext, verbose: bool, no_banner: bool, plain: bool) {
    terminal::init(ctx, verbose, no_banner, plain);
}

/// Restores terminal output state before the binary exits.
pub fn finish_output(success: bool, summary: &str) {
    terminal::finish(success, summary);
}
