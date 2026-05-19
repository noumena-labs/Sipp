//! Backend selection (CPU/CUDA/Metal/Vulkan) with capability fall-back.

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
        let capabilities = capabilities.normalized();
        let requested = options.backend;
        let selected = select_backend(requested, &capabilities)?;
        let mut config = options.runtime.clone();
        apply_stats_mode(&mut config, options.stats);
        apply_backend_layers(&mut config, requested, &selected);

        Ok(BackendPlan {
            selection: BackendSelection {
                requested,
                selected: selected.clone(),
                available: capabilities.available,
                gpu_offload_expected: selected != "cpu"
                    && config.placement.gpu_layers != GpuLayerConfig::Count(0),
                reason: Some(selection_reason(requested, &selected)),
            },
            config,
        })
    }
}

impl BackendCapabilities {
    fn normalized(&self) -> Self {
        let mut compiled = normalize_backend_names(&self.compiled);
        let mut available = normalize_backend_names(&self.available);
        if available.is_empty() {
            available.push("cpu".to_string());
        }
        if compiled.is_empty() && available.iter().any(|name| name == "cpu") {
            compiled.push("cpu".to_string());
        }

        Self {
            compiled,
            available,
            gpu_offload_supported: self.gpu_offload_supported,
        }
    }
}

pub fn read_backend_capabilities() -> Result<BackendCapabilities, ModelError> {
    let raw = backend_observability_json(true).map_err(ModelError::from)?;
    let value = serde_json::from_str::<Value>(&raw)?;

    let mut compiled = value
        .get("compiled")
        .and_then(Value::as_object)
        .map_or_else(Vec::new, |map| Vec::with_capacity(map.len()));
    if let Some(map) = value.get("compiled").and_then(Value::as_object) {
        for (name, enabled) in map {
            if enabled.as_bool().unwrap_or(false) {
                let normalized = normalize_backend_name(name);
                if !normalized.is_empty() {
                    compiled.push(normalized);
                }
            }
        }
    }
    compiled.sort();
    compiled.dedup();

    let mut available = value
        .get("availableBackends")
        .and_then(Value::as_array)
        .map(|items| {
            let mut available = Vec::with_capacity(items.len());
            available.extend(
                items
                    .iter()
                    .filter_map(|item| item.get("name").and_then(Value::as_str))
                    .map(normalize_backend_name)
                    .filter(|name| !name.is_empty()),
            );
            available
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

fn normalize_backend_names(names: &[String]) -> Vec<String> {
    let mut normalized = Vec::with_capacity(names.len());
    normalized.extend(
        names
            .iter()
            .map(|name| normalize_backend_name(name))
            .filter(|name| !name.is_empty()),
    );
    normalized.sort();
    normalized.dedup();
    normalized
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
mod tests;
