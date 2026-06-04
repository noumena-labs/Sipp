use serde::{Deserialize, Serialize};

use crate::engine::protocol::{BackendInfo, EngineStatus, RequestState};
use crate::engine::{EngineStats, NativeRuntimeConfig, ResolvedRuntimeLimits};

use super::model::ModelInfo;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/lifecycle/types/runtime_tests.rs"]
mod runtime_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatsMode {
    Off,
    #[default]
    Basic,
    Profile,
}

impl StatsMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Basic => "basic",
            Self::Profile => "profile",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendPreference {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
    Vulkan,
    WebGpu,
}

impl BackendPreference {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::WebGpu => "webgpu",
        }
    }
}

pub const DEFAULT_MODEL_BACKEND: &str = BackendPreference::Auto.as_str();
pub const DEFAULT_MODEL_STATS: &str = StatsMode::Basic.as_str();

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendSelection {
    pub requested: BackendPreference,
    pub selected: String,
    pub available: Vec<String>,
    pub gpu_offload_expected: bool,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelLoadOptions {
    pub backend: BackendPreference,
    pub stats: StatsMode,
    pub runtime: NativeRuntimeConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelServiceState {
    pub status: EngineStatus,
    pub model: Option<ModelInfo>,
    pub backend: BackendInfo,
    pub runtime: Option<ResolvedRuntimeLimits>,
    pub requests: Vec<RequestState>,
    pub stats: EngineStats,
    pub updated_at_unix_ms: u64,
}

impl Default for ModelServiceState {
    fn default() -> Self {
        Self {
            status: EngineStatus::Idle,
            model: None,
            backend: BackendInfo::default(),
            runtime: None,
            requests: Vec::new(),
            stats: EngineStats::default(),
            updated_at_unix_ms: 0,
        }
    }
}
