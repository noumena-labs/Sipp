//! Backend selection (CPU/CUDA/Metal/Vulkan) with capability fall-back.

use serde_json::Value;

use crate::backend::{
    backend_observability_json, json_array_strings, json_bool, KEY_AVAILABLE_BACKENDS,
    KEY_COMPILED, KEY_GPU_OFFLOAD_SUPPORTED, KEY_NAME,
};
use crate::collection::sorted_unique_non_empty_strings;
use crate::engine::{GpuLayerConfig, NativeRuntimeConfig};

use super::util::invalid_source;
use super::{BackendPreference, BackendSelection, ModelError, ModelLoadOptions, StatsMode};

const CPU_BACKEND: &str = BackendPreference::Cpu.as_str();
const AUTO_BACKEND_CANDIDATES: [BackendPreference; 4] = [
    BackendPreference::Cuda,
    BackendPreference::Metal,
    BackendPreference::Vulkan,
    BackendPreference::WebGpu,
];

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
                gpu_offload_expected: gpu_offload_expected(&selected, &config),
                reason: Some(selection_reason(requested, &selected)),
            },
            config,
        })
    }
}

impl BackendCapabilities {
    fn normalized(&self) -> Self {
        let mut compiled = normalize_backend_names(&self.compiled);
        let available = normalize_backend_names_or_cpu(&self.available);
        if compiled.is_empty() && contains_cpu_backend(&available) {
            compiled.push(cpu_backend_name());
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

    let compiled = value
        .get(KEY_COMPILED)
        .and_then(Value::as_object)
        .map(|map| {
            normalize_backend_values(
                map.iter()
                    .filter(|(_, enabled)| enabled.as_bool().unwrap_or(false))
                    .map(|(name, _)| name.as_str()),
            )
        })
        .unwrap_or_default();

    let available = normalize_backend_names_or_cpu(&json_array_strings(
        &value,
        KEY_AVAILABLE_BACKENDS,
        KEY_NAME,
    ));

    Ok(BackendCapabilities {
        compiled,
        available,
        gpu_offload_supported: json_bool(&value, KEY_GPU_OFFLOAD_SUPPORTED).unwrap_or(false),
    })
}

fn normalize_backend_names(names: &[String]) -> Vec<String> {
    normalize_backend_values(names.iter().map(String::as_str))
}

fn normalize_backend_names_or_cpu(names: &[String]) -> Vec<String> {
    with_cpu_fallback(normalize_backend_names(names))
}

fn normalize_backend_values<'a>(names: impl Iterator<Item = &'a str>) -> Vec<String> {
    sorted_unique_non_empty_strings(names.map(normalize_backend_name))
}

fn with_cpu_fallback(mut names: Vec<String>) -> Vec<String> {
    if names.is_empty() {
        names.push(cpu_backend_name());
    }
    names
}

fn select_backend(
    requested: BackendPreference,
    capabilities: &BackendCapabilities,
) -> Result<String, ModelError> {
    match requested {
        BackendPreference::Auto => Ok(select_auto_backend(capabilities)),
        BackendPreference::Cpu => Ok(cpu_backend_name()),
        BackendPreference::Cuda
        | BackendPreference::Metal
        | BackendPreference::Vulkan
        | BackendPreference::WebGpu => require_backend(requested.as_str(), capabilities),
    }
}

fn select_auto_backend(capabilities: &BackendCapabilities) -> String {
    for candidate in AUTO_BACKEND_CANDIDATES {
        let name = candidate.as_str();
        if backend_is_usable(name, capabilities) {
            return owned_backend_name(name);
        }
    }
    cpu_backend_name()
}

fn require_backend(
    name: &'static str,
    capabilities: &BackendCapabilities,
) -> Result<String, ModelError> {
    if backend_is_usable(name, capabilities) {
        Ok(owned_backend_name(name))
    } else {
        Err(backend_unavailable(name, &capabilities.available))
    }
}

fn backend_unavailable(name: &str, available: &[String]) -> ModelError {
    invalid_source(format!(
        "requested backend {name} is not compiled or not available; available backends: {}",
        available.join(", ")
    ))
}

fn backend_is_usable(name: &str, capabilities: &BackendCapabilities) -> bool {
    if is_cpu_backend(name) {
        return true;
    }
    capabilities.gpu_offload_supported
        && contains_backend(&capabilities.compiled, name)
        && contains_backend(&capabilities.available, name)
}

fn contains_backend(names: &[String], name: &str) -> bool {
    names.iter().any(|item| item == name)
}

fn contains_cpu_backend(names: &[String]) -> bool {
    names.iter().any(|name| is_cpu_backend(name))
}

fn apply_stats_mode(config: &mut NativeRuntimeConfig, stats: StatsMode) {
    let (runtime_metrics, backend_profiling) = stats_mode_observability(stats);
    config.observability.runtime_metrics = runtime_metrics;
    config.observability.backend_profiling = backend_profiling;
}

fn stats_mode_observability(stats: StatsMode) -> (bool, bool) {
    match stats {
        StatsMode::Off => (false, false),
        StatsMode::Basic => (true, false),
        StatsMode::Profile => (true, true),
    }
}

fn apply_backend_layers(
    config: &mut NativeRuntimeConfig,
    requested: BackendPreference,
    selected: &str,
) {
    if requested == BackendPreference::Cpu || is_cpu_backend(selected) {
        config.placement.gpu_layers = GpuLayerConfig::Count(0);
    }
}

fn gpu_offload_expected(selected: &str, config: &NativeRuntimeConfig) -> bool {
    !is_cpu_backend(selected) && config.placement.gpu_layers != GpuLayerConfig::Count(0)
}

fn selection_reason(requested: BackendPreference, selected: &str) -> String {
    match requested {
        BackendPreference::Auto => format!("auto selected {selected}"),
        _ => format!("{} requested", requested.as_str()),
    }
}

fn normalize_backend_name(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    if lower.contains("cuda") {
        owned_backend_name(BackendPreference::Cuda.as_str())
    } else if lower.contains("metal") {
        owned_backend_name(BackendPreference::Metal.as_str())
    } else if lower.contains("vulkan") {
        owned_backend_name(BackendPreference::Vulkan.as_str())
    } else if lower.contains("webgpu") {
        owned_backend_name(BackendPreference::WebGpu.as_str())
    } else if lower.contains("cpu") {
        cpu_backend_name()
    } else {
        lower
    }
}

fn is_cpu_backend(name: &str) -> bool {
    name == CPU_BACKEND
}

fn cpu_backend_name() -> String {
    owned_backend_name(CPU_BACKEND)
}

fn owned_backend_name(name: &str) -> String {
    name.to_string()
}

#[cfg(test)]
mod tests {
    mod backend_policy_tests;
}
