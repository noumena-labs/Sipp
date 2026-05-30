//! Build orchestration support for the CogentLM workspace.
//!
//! The binary entrypoint parses CLI commands and delegates to these modules so
//! target-specific build logic and host toolchain setup stay isolated.

pub mod cli;
pub mod clean;
pub mod doctor;
pub(crate) mod launcher;
pub(crate) mod output;
pub mod run;
pub mod setup;
pub(crate) mod splash;
pub mod targets;
pub mod toolchain;
pub mod toolchains;
pub mod utils;

/// Configures terminal output for the xtask binary entrypoint.
pub fn configure_output(ctx: &utils::BuildContext, verbose: bool, no_banner: bool, plain: bool) {
    output::init(ctx, verbose, no_banner, plain);
}

/// Restores terminal output state before the binary exits.
pub fn finish_output() {
    output::finish();
}
