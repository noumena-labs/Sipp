//! Unit tests for the parent module.

use super::super::*;
use crate::lifecycle::test_support::strings;
use crate::runtime::config::SplitMode;

fn caps(compiled: &[&str], available: &[&str]) -> BackendCapabilities {
    BackendCapabilities {
        compiled: strings(compiled),
        available: strings(available),
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

    let plan =
        BackendPolicy::select_with_capabilities(&options, &caps(&[], &["cpu"])).expect("backend");

    assert_eq!(plan.selection.selected, "cpu");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Count(0));
    assert_eq!(plan.config.placement.split_mode, SplitMode::Layer);
    assert_eq!(
        plan.config.context.flash_attention,
        FlashAttentionMode::Disabled
    );
    assert!(!plan.config.context.offload_kqv);
    assert!(!plan.config.context.op_offload);
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
fn webgpu_selects_full_offload_by_default() {
    let options = ModelLoadOptions {
        backend: BackendPreference::WebGpu,
        ..ModelLoadOptions::default()
    };

    let plan =
        BackendPolicy::select_with_capabilities(&options, &caps(&["webgpu"], &["cpu", "webgpu"]))
            .expect("webgpu");

    assert_eq!(plan.selection.selected, "webgpu");
    assert_eq!(plan.config.placement.gpu_layers, GpuLayerConfig::Auto);
    assert!(plan.selection.gpu_offload_expected);
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
    assert_eq!(plan.config.placement.split_mode, SplitMode::Layer);
    assert!(!plan.config.context.offload_kqv);
}

#[test]
fn select_with_capabilities_normalizes_backend_names() {
    let plan = BackendPolicy::select_with_capabilities(
        &ModelLoadOptions {
            backend: BackendPreference::Cuda,
            ..ModelLoadOptions::default()
        },
        &BackendCapabilities {
            compiled: strings(&["CUDA backend", "cuda"]),
            available: strings(&["NVIDIA CUDA", "CPU"]),
            gpu_offload_supported: true,
        },
    )
    .expect("cuda");

    assert_eq!(plan.selection.selected, "cuda");
    assert_eq!(plan.selection.available, vec!["cpu", "cuda"]);
}

#[test]
fn normalize_backend_names_drops_empty_and_deduplicates() {
    let names = strings(&[" CUDA backend ", "cuda", " ", "CPU"]);

    assert_eq!(normalize_backend_names(&names), vec!["cpu", "cuda"]);
    assert_eq!(normalize_backend_names_or_cpu(&[]), vec!["cpu"]);
}
