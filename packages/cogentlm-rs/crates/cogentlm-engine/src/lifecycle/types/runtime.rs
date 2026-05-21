use serde::{Deserialize, Serialize};

use crate::engine::protocol::{BackendInfo, EngineStatus, RequestState};
use crate::engine::{EngineStats, NativeRuntimeConfig, ResolvedRuntimeLimits};

use super::model::ModelInfo;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StatsMode {
    Off,
    #[default]
    Basic,
    Profile,
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
