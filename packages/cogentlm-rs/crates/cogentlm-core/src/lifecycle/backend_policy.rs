use serde_json::Value;

use crate::backend::backend_observability_json;
use crate::engine::{GpuLayerConfig, NativeRuntimeConfig};

use super::{BackendPreference, BackendSelection, ModelError, ModelLoadOptions, StatsMode};

#[derive(Debug, Clone, PartialEq)]
pub struct BackendPlan {
    pub config: NativeRuntimeConfig,
    pub selection: BackendSelection,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BackendCapabilities {
    pub compiled: Vec<String>,
    pub available: Vec<String>,
    pub gpu_offload_supported: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct BackendPolicy;

impl BackendPolicy {
    pub fn select(options: &ModelLoadOptions) -> Result<BackendPlan, ModelError> {
        Self::select_with_capabilities(options, &read_backend_capabilities()?)
    }

    pub fn select_with_capabilities(
        options: &ModelLoadOptions,
        capabilities: &BackendCapabilities,
    ) -> Result<BackendPlan, ModelError> {
        let requested = options.backend;
        let selected = select_backend(requested, capabilities)?;
        let mut config = options.runtime.clone();
        apply_stats_mode(&mut config, options.stats);
        apply_backend_layers(&mut config, requested, &selected);

        Ok(BackendPlan {
            selection: BackendSelection {
                requested,
                selected: selected.clone(),
                available: capabilities.available.clone(),
                gpu_offload_expected: selected != "cpu"
                    && config.placement.gpu_layers != GpuLayerConfig::Count(0),
                reason: Some(selection_reason(requested, &selected)),
            },
            config,
        })
    }
}

pub fn read_backend_capabilities() -> Result<BackendCapabilities, ModelError> {
    let raw = backend_observability_json(true).map_err(ModelError::from)?;
    let value = serde_json::from_str::<Value>(&raw)?;

    let mut compiled = Vec::new();
    if let Some(map) = value.get("compiled").and_then(Value::as_object) {
        for (name, enabled) in map {
            if enabled.as_bool().unwrap_or(false) {
                compiled.push(normalize_backend_name(name));
            }
        }
    }
    compiled.sort();
    compiled.dedup();

    let mut available = value
        .get("availableBackends")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("name").and_then(Value::as_str))
                .map(normalize_backend_name)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if available.is_empty() {
        available.push("cpu".to_string());
    }
    available.sort();
    available.dedup();

    Ok(BackendCapabilities {
        compiled,
        available,
        gpu_offload_supported: value
            .get("gpuOffloadSupported")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn select_backend(
    requested: BackendPreference,
    capabilities: &BackendCapabilities,
) -> Result<String, ModelError> {
    match requested {
        BackendPreference::Auto => Ok(select_auto_backend(capabilities)),
        BackendPreference::Cpu => Ok("cpu".to_string()),
        BackendPreference::Cuda => require_backend("cuda", capabilities),
        BackendPreference::Metal => require_backend("metal", capabilities),
        BackendPreference::Vulkan => require_backend("vulkan", capabilities),
        BackendPreference::WebGpu => require_backend("webgpu", capabilities),
    }
}

fn select_auto_backend(capabilities: &BackendCapabilities) -> String {
    for candidate in ["cuda", "metal", "vulkan", "webgpu"] {
        if backend_is_usable(candidate, capabilities) {
            return candidate.to_string();
        }
    }
    "cpu".to_string()
}

fn require_backend(
    name: &'static str,
    capabilities: &BackendCapabilities,
) -> Result<String, ModelError> {
    if backend_is_usable(name, capabilities) {
        Ok(name.to_string())
    } else {
        Err(ModelError::InvalidModelSource(format!(
            "requested backend {name} is not compiled or not available; available backends: {}",
            capabilities.available.join(", ")
        )))
    }
}

fn backend_is_usable(name: &str, capabilities: &BackendCapabilities) -> bool {
    if name == "cpu" {
        return true;
    }
    capabilities.gpu_offload_supported
        && capabilities.compiled.iter().any(|item| item == name)
        && capabilities.available.iter().any(|item| item == name)
}

fn apply_stats_mode(config: &mut NativeRuntimeConfig, stats: StatsMode) {
    match stats {
        StatsMode::Off => {
            config.observability.runtime_metrics = false;
            config.observability.backend_profiling = false;
        }
        StatsMode::Basic => {
            config.observability.runtime_metrics = true;
            config.observability.backend_profiling = false;
        }
        StatsMode::Profile => {
            config.observability.runtime_metrics = true;
            config.observability.backend_profiling = true;
        }
    }
}

fn apply_backend_layers(
    config: &mut NativeRuntimeConfig,
    requested: BackendPreference,
    selected: &str,
) {
    if requested == BackendPreference::Cpu || selected == "cpu" {
        config.placement.gpu_layers = GpuLayerConfig::Count(0);
    } else if config.placement.gpu_layers == GpuLayerConfig::Auto {
        config.placement.gpu_layers = GpuLayerConfig::Auto;
    }
}

fn selection_reason(requested: BackendPreference, selected: &str) -> String {
    match requested {
        BackendPreference::Auto => format!("auto selected {selected}"),
        BackendPreference::Cpu => "cpu requested".to_string(),
        BackendPreference::Cuda => "cuda requested".to_string(),
        BackendPreference::Metal => "metal requested".to_string(),
        BackendPreference::Vulkan => "vulkan requested".to_string(),
        BackendPreference::WebGpu => "webgpu requested".to_string(),
    }
}

fn normalize_backend_name(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    if lower.contains("cuda") {
        "cuda".to_string()
    } else if lower.contains("metal") {
        "metal".to_string()
    } else if lower.contains("vulkan") {
        "vulkan".to_string()
    } else if lower.contains("webgpu") {
        "webgpu".to_string()
    } else if lower.contains("cpu") {
        "cpu".to_string()
    } else {
        lower
    }
}

#[cfg(test)]
mod tests {
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
}
