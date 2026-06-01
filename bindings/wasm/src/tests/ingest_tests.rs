//! Unit tests for the parent module.

use super::*;

#[test]
fn cache_layout_uses_split_for_unknown_or_large_sources() {
    assert_eq!(browser_cache_layout(1024, true, 2048, 512), 0);
    assert_eq!(browser_cache_layout(4096, true, 2048, 512), 1);
    assert_eq!(browser_cache_layout(0, false, 2048, 512), 1);
}
