//! Tests the `toolchains::bun` module in `xtask`.
//!
//! Covers managed Bun version detection using small fake executables instead of
//! downloading Bun from GitHub.

#[cfg(unix)]
use crate::test_support::TempDir;

use super::{bun_version_matches, BUN_VERSION};

#[cfg(unix)]
#[test]
fn bun_version_matching_requires_pinned_version() {
    use std::os::unix::fs::PermissionsExt;

    let temp = TempDir::new("bun-version");
    let old_bun = temp.write("old-bun", "#!/usr/bin/env sh\necho 0.0.0\n");
    let current_bun = temp.write(
        "current-bun",
        format!("#!/usr/bin/env sh\necho {BUN_VERSION}\n"),
    );
    let mut permissions = std::fs::metadata(&old_bun).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&old_bun, permissions.clone()).unwrap();
    std::fs::set_permissions(&current_bun, permissions).unwrap();

    assert!(!bun_version_matches(&old_bun));
    assert!(bun_version_matches(&current_bun));
}

#[test]
fn missing_bun_does_not_match_pinned_version() {
    assert!(!bun_version_matches(std::path::Path::new(
        "sipp-missing-bun"
    )));
}
