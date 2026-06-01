//! Unit tests for the parent module.

use super::*;

#[test]
fn local_asset_source_requires_source_path() {
    let error = serde_json::from_str::<AssetSource>(r#"{"kind":"local"}"#)
        .expect_err("local source without path should be rejected");

    assert!(error.to_string().contains("missing field `path`"));
}
