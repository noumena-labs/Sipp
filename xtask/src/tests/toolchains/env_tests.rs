//! Tests the `toolchains::env` module in `xtask`.
//!
//! Covers deterministic platform path-separator selection without constructing
//! commands that would bootstrap or inspect external toolchains.

use super::path_separator;

#[test]
fn path_separator_matches_host_platform() {
    assert_eq!(path_separator(), if cfg!(windows) { ";" } else { ":" });
}
