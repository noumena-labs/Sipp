//! Tests the `lifecycle::types::assets` module in `cogentlm-engine`.
//!
//! Covers lifecycle registry, storage, browser, service, and pairing behavior with temporary storage and pure fixtures instead of native runtime loading.

use super::*;

#[test]
fn local_asset_source_requires_source_path() {
    let error = serde_json::from_str::<AssetSource>(r#"{"kind":"local"}"#)
        .expect_err("local source without path should be rejected");

    assert!(error.to_string().contains("missing field `path`"));
}
