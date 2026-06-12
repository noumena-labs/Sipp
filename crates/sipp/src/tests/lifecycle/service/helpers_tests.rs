//! Tests the `lifecycle::service::helpers` module in `sipp`.
//!
//! Covers path comparison, stable model IDs, and runtime fingerprinting using
//! pure lifecycle/config values without native runtime loading.

use std::path::Path;

use crate::lifecycle::{
    BackendPreference, BackendSelection, ModelModality, ModelStatus, PairingPlan,
};
use crate::runtime::config::{GpuLayerConfig, ModelPlacementConfig, NativeRuntimeConfig};

use super::*;

fn plan(projector_asset_id: Option<&str>) -> PairingPlan {
    PairingPlan {
        model_asset_ids: vec!["asset-b".to_string(), "asset-a".to_string()],
        projector_asset_id: projector_asset_id.map(str::to_string),
        name: "model".to_string(),
        modality: ModelModality::Vision,
        status: ModelStatus::Ready,
        compatible_vision_projector_types: Vec::new(),
    }
}

fn backend_plan(selected: &str, config: NativeRuntimeConfig) -> BackendPlan {
    BackendPlan {
        config,
        selection: BackendSelection {
            requested: BackendPreference::Auto,
            selected: selected.to_string(),
            available: vec!["cpu".to_string(), selected.to_string()],
            gpu_offload_expected: selected != "cpu",
            reason: None,
        },
    }
}

#[test]
fn same_path_matches_platform_case_rules() {
    #[cfg(windows)]
    assert!(same_path(
        Path::new("C:/Models/Test.GGUF"),
        Path::new("c:/models/test.gguf")
    ));

    #[cfg(not(windows))]
    assert!(!same_path(
        Path::new("/models/Test.GGUF"),
        Path::new("/models/test.gguf")
    ));
}

#[test]
fn model_id_from_plan_sorts_assets_and_includes_projector() {
    let left = plan(Some("asset-c"));
    let right = PairingPlan {
        model_asset_ids: vec!["asset-a".to_string(), "asset-b".to_string()],
        ..left.clone()
    };
    let without_projector = plan(None);

    assert_eq!(model_id_from_plan(&left), model_id_from_plan(&right));
    assert_ne!(
        model_id_from_plan(&left),
        model_id_from_plan(&without_projector)
    );
    assert!(model_id_from_plan(&left).starts_with("model-"));
}

#[test]
fn runtime_fingerprint_changes_with_runtime_backend_or_assets() {
    let entry = crate::lifecycle::model_entry_from_assets("model-a", "model", &plan(None));
    let cpu = backend_plan("cpu", NativeRuntimeConfig::default());
    let mut gpu_config = NativeRuntimeConfig::default();
    gpu_config.placement = ModelPlacementConfig {
        gpu_layers: GpuLayerConfig::Auto,
        ..ModelPlacementConfig::default()
    };
    let gpu = backend_plan("cuda", gpu_config);
    let mut changed_entry = entry.clone();
    changed_entry.model_asset_ids.push("asset-c".to_string());

    let cpu_fingerprint = runtime_fingerprint(&entry, &cpu).expect("fingerprint");

    assert_ne!(
        cpu_fingerprint,
        runtime_fingerprint(&entry, &gpu).expect("backend fingerprint")
    );
    assert_ne!(
        cpu_fingerprint,
        runtime_fingerprint(&changed_entry, &cpu).expect("asset fingerprint")
    );
}
