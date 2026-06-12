//! Tests the `cogentlm` crate root public facade helpers.
//!
//! Covers deterministic package metadata without loading local models or calling
//! gateway endpoints.

use super::package_version;

#[test]
fn package_version_matches_manifest_version() {
    assert_eq!(package_version(), env!("CARGO_PKG_VERSION"));
}
