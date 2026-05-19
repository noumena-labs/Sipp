//! Unit tests for the parent module.

use super::*;

fn caps(compiled: &[&str], available: &[&str]) -> BackendCapabilities {
    BackendCapabilities {
        compiled: compiled.iter().map(|value| (*value).to_string()).collect(),
        available: available.iter().map(|value| (*value).to_string()).collect(),
        gpu_offload_supported: compiled.iter().any(|value| *value != "cpu"),
    }
}

#[test]
fn cpu_forces_gpu_layers_zero() {
    let options = ModelLoadOptions {
        backend: BackendPreference::Cpu,
        stats: StatsMode::Off,
        ..ModelLoadOptions::default()
    };

    let plan = BackendPolicy::select_with_capabilities(&options, &caps(&[], &["cpu"]))
        .expect("backend");

    assert_eq!(plan.selection.selected, "cpu");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Count(0));
    assert!(!plan.config.observability.runtime_metrics);
    assert!(!plan.config.observability.backend_profiling);
}

#[test]
fn cuda_requires_compiled_available_backend() {
    let options = ModelLoadOptions {
        backend: BackendPreference::Cuda,
        ..ModelLoadOptions::default()
    };

    let error = BackendPolicy::select_with_capabilities(&options, &caps(&[], &["cpu"]))
        .expect_err("missing cuda");

    assert!(matches!(error, ModelError::InvalidModelSource(_)));
}

#[test]
fn cuda_selects_full_offload_by_default() {
    let options = ModelLoadOptions {
        backend: BackendPreference::Cuda,
        stats: StatsMode::Profile,
        ..ModelLoadOptions::default()
    };

    let plan =
        BackendPolicy::select_with_capabilities(&options, &caps(&["cuda"], &["cuda", "cpu"]))
            .expect("cuda");

    assert_eq!(plan.selection.selected, "cuda");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Auto);
    assert!(plan.config.observability.runtime_metrics);
    assert!(plan.config.observability.backend_profiling);
}

#[test]
fn auto_prefers_accelerator_then_cpu() {
    let plan = BackendPolicy::select_with_capabilities(
        &ModelLoadOptions::default(),
        &caps(&["vulkan"], &["cpu", "vulkan"]),
    )
    .expect("auto");
    assert_eq!(plan.selection.selected, "vulkan");

    let plan =
        BackendPolicy::select_with_capabilities(&ModelLoadOptions::default(), &caps(&[], &[]))
            .expect("auto cpu");
    assert_eq!(plan.selection.selected, "cpu");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Count(0));
}

#[test]
fn select_with_capabilities_normalizes_backend_names() {
    let plan = BackendPolicy::select_with_capabilities(
        &ModelLoadOptions {
            backend: BackendPreference::Cuda,
            ..ModelLoadOptions::default()
        },
        &BackendCapabilities {
            compiled: vec!["CUDA backend".to_string(), "cuda".to_string()],
            available: vec!["NVIDIA CUDA".to_string(), "CPU".to_string()],
            gpu_offload_supported: true,
        },
    )
    .expect("cuda");

    assert_eq!(plan.selection.selected, "cuda");
    assert_eq!(plan.selection.available, vec!["cpu", "cuda"]);
}

#[test]
fn normalize_backend_names_drops_empty_and_deduplicates() {
    let names = vec![
        " CUDA backend ".to_string(),
        "cuda".to_string(),
        " ".to_string(),
        "CPU".to_string(),
    ];

    assert_eq!(normalize_backend_names(&names), vec!["cpu", "cuda"]);
}
