//! Unit tests for the parent module.

use super::super::*;

#[test]
fn local_asset_source_requires_source_path() {
    let error = serde_json::from_str::<AssetSource>(r#"{"kind":"local"}"#)
        .expect_err("local source without path should be rejected");

    assert!(error.to_string().contains("missing field `path`"));
}

#[test]
fn runtime_choice_helpers_accept_binding_spellings() {
    assert_eq!(DEFAULT_MODEL_BACKEND, "auto");
    assert_eq!(DEFAULT_MODEL_STATS, "basic");
    assert_eq!(
        BackendPreference::from_choice("web-gpu"),
        Some(BackendPreference::WebGpu)
    );
    assert_eq!(
        BackendPreference::from_choice("web gpu"),
        Some(BackendPreference::WebGpu)
    );
    assert_eq!(StatsMode::from_choice("profile"), Some(StatsMode::Profile));
    assert_eq!(BackendPreference::Cuda.as_str(), "cuda");
    assert_eq!(StatsMode::Off.as_str(), "off");
}
