//! Tests the `lifecycle::types::runtime` module in `cogentlm-engine`.
//!
//! Covers model runtime option defaults, enum wire names, and idle service
//! state values with deterministic pure fixtures.

use super::*;
use crate::engine::protocol::EngineStatus;
use serde_json::json;

#[test]
fn stats_mode_and_backend_preference_strings_cover_all_variants() {
    assert_eq!(StatsMode::Off.as_str(), "off");
    assert_eq!(StatsMode::Basic.as_str(), "basic");
    assert_eq!(StatsMode::Profile.as_str(), "profile");
    assert_eq!(BackendPreference::Auto.as_str(), "auto");
    assert_eq!(BackendPreference::Cpu.as_str(), "cpu");
    assert_eq!(BackendPreference::Cuda.as_str(), "cuda");
    assert_eq!(BackendPreference::Metal.as_str(), "metal");
    assert_eq!(BackendPreference::Vulkan.as_str(), "vulkan");
    assert_eq!(BackendPreference::WebGpu.as_str(), "webgpu");
    assert_eq!(DEFAULT_MODEL_BACKEND, "auto");
    assert_eq!(DEFAULT_MODEL_STATS, "basic");
}

#[test]
fn runtime_option_defaults_are_stable() {
    assert_eq!(StatsMode::default(), StatsMode::Basic);
    assert_eq!(BackendPreference::default(), BackendPreference::Auto);

    let options = ModelLoadOptions::default();
    assert_eq!(options.backend, BackendPreference::Auto);
    assert_eq!(options.stats, StatsMode::Basic);
    assert_eq!(options.runtime, NativeRuntimeConfig::default());
}

#[test]
fn runtime_enums_round_trip_through_snake_case_serde() {
    assert_eq!(
        serde_json::from_value::<StatsMode>(json!("profile")).expect("stats mode"),
        StatsMode::Profile
    );
    assert_eq!(
        serde_json::to_value(StatsMode::Off).expect("stats mode"),
        "off"
    );

    let cases = [
        (BackendPreference::Auto, "auto"),
        (BackendPreference::Cpu, "cpu"),
        (BackendPreference::Cuda, "cuda"),
        (BackendPreference::Metal, "metal"),
        (BackendPreference::Vulkan, "vulkan"),
        (BackendPreference::WebGpu, "web_gpu"),
    ];
    for (backend, wire) in cases {
        assert_eq!(serde_json::to_value(backend).expect("backend"), wire);
        assert_eq!(
            serde_json::from_value::<BackendPreference>(json!(wire)).expect("backend"),
            backend
        );
    }
}

#[test]
fn backend_selection_default_and_camel_case_serde_are_stable() {
    let selection = BackendSelection::default();

    assert_eq!(selection.requested, BackendPreference::Auto);
    assert_eq!(selection.selected, "");
    assert!(selection.available.is_empty());
    assert!(!selection.gpu_offload_expected);
    assert!(selection.reason.is_none());

    let value = serde_json::to_value(BackendSelection {
        requested: BackendPreference::Vulkan,
        selected: "cpu".to_string(),
        available: vec!["cpu".to_string(), "vulkan".to_string()],
        gpu_offload_expected: true,
        reason: Some("fallback".to_string()),
    })
    .expect("backend selection");

    assert_eq!(value["requested"], "vulkan");
    assert_eq!(value["selected"], "cpu");
    assert_eq!(value["available"], json!(["cpu", "vulkan"]));
    assert_eq!(value["gpuOffloadExpected"], true);
    assert_eq!(value["reason"], "fallback");
}

#[test]
fn model_service_state_default_reports_idle_empty_state() {
    let state = ModelServiceState::default();

    assert_eq!(state.status, EngineStatus::Idle);
    assert!(state.model.is_none());
    assert!(state.runtime.is_none());
    assert!(state.requests.is_empty());
    assert_eq!(state.updated_at_unix_ms, 0);
}
