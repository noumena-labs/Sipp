//! NativeRuntimeConfig and friends: typed runtime configuration plus serialization to the shim JSON shape.

use std::fmt::{Display, Write as _};

use serde::{Deserialize, Serialize};

mod context;
mod placement;
mod sampling;

use super::SchedulerPolicyConfig;
#[cfg(test)]
use super::SchedulerPolicyMode;
pub use context::{ContextRuntimeConfig, FlashAttentionMode, KvCacheType, RopeScaling};
pub use placement::{GpuLayerConfig, ModelPlacementConfig, SplitMode};
use sampling::merge_sampling_override_json;
pub use sampling::{LogitBias, SamplerStage, SamplingRuntimeConfig};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct NativeRuntimeConfig {
    pub placement: ModelPlacementConfig,
    pub context: ContextRuntimeConfig,
    pub sampling: SamplingRuntimeConfig,
    pub scheduler: SchedulerRuntimeConfig,
    pub cache: CacheRuntimeConfig,
    pub multimodal: MultimodalRuntimeConfig,
    pub residency: ResidencyRuntimeConfig,
    pub observability: ObservabilityRuntimeConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ResolvedRuntimeLimits {
    pub n_ctx: i32,
    pub n_batch: i32,
    pub n_ubatch: i32,
    pub n_parallel: i32,
    pub kv_unified: bool,
    pub flash_attention: String,
    pub cache_type_k: String,
    pub cache_type_v: String,
}

impl NativeRuntimeConfig {
    pub fn normalize(mut self) -> Self {
        self.context.normalize();
        self.scheduler.normalize();
        self.cache.normalize();
        self.multimodal.normalize();
        self.residency.normalize();
        self
    }

    pub fn max_sequences(&self) -> i32 {
        self.context.n_parallel.unwrap_or(1).max(1)
    }

    pub fn llama_common_args(&self) -> Vec<String> {
        let normalized = self.clone().normalize();
        let mut args =
            Vec::with_capacity(normalized.placement.arg_len() + normalized.context.arg_len());
        normalized.placement.push_args(&mut args);
        normalized.context.push_args(&mut args);
        args
    }

    pub fn sampling_json(&self) -> String {
        self.sampling_json_with_override(None)
    }

    pub fn try_sampling_json(&self) -> serde_json::Result<String> {
        self.try_sampling_json_with_override(None)
    }

    pub fn sampling_json_with_override(
        &self,
        override_config: Option<&SamplingRuntimeConfig>,
    ) -> String {
        self.try_sampling_json_with_override(override_config)
            .unwrap_or_else(|_| "{}".to_string())
    }

    pub fn try_sampling_json_with_override(
        &self,
        override_config: Option<&SamplingRuntimeConfig>,
    ) -> serde_json::Result<String> {
        let mut value = serde_json::to_value(&self.sampling)?;
        if let Some(override_config) = override_config {
            let override_value = serde_json::to_value(override_config)?;
            merge_sampling_override_json(&mut value, override_value);
        }
        serde_json::to_string(&value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct SchedulerRuntimeConfig {
    pub continuous_batching: bool,
    pub policy: SchedulerPolicyConfig,
    pub prefill_chunk_size: i32,
    pub max_running_requests: Option<i32>,
    pub max_queued_requests: Option<i32>,
}

impl Default for SchedulerRuntimeConfig {
    fn default() -> Self {
        Self {
            continuous_batching: true,
            policy: SchedulerPolicyConfig::default(),
            prefill_chunk_size: 0,
            max_running_requests: None,
            max_queued_requests: None,
        }
    }
}

impl SchedulerRuntimeConfig {
    fn normalize(&mut self) {
        self.policy.decode_token_reserve = self.policy.decode_token_reserve.max(0);
        self.prefill_chunk_size = self.prefill_chunk_size.max(0);
        self.max_running_requests = positive_or_none(self.max_running_requests, 1);
        self.max_queued_requests = positive_or_none(self.max_queued_requests, 1);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CacheRuntimeConfig {
    pub mode: KvReuseMode,
    pub retained_prefix_tokens: i32,
    pub snapshot_interval_tokens: i32,
    pub max_snapshot_entries: i32,
    pub max_snapshot_bytes: usize,
    pub max_session_entries: i32,
    pub cache_key_policy: CacheKeyPolicy,
    pub enable_context_checkpoints: bool,
    pub checkpoint_every_tokens: i32,
}

impl Default for CacheRuntimeConfig {
    fn default() -> Self {
        Self {
            mode: KvReuseMode::LiveSlotPrefix,
            retained_prefix_tokens: 100,
            snapshot_interval_tokens: 128,
            max_snapshot_entries: 32,
            max_snapshot_bytes: 256 * 1024 * 1024,
            max_session_entries: 8,
            cache_key_policy: CacheKeyPolicy::ContextKey,
            enable_context_checkpoints: false,
            checkpoint_every_tokens: 8192,
        }
    }
}

impl CacheRuntimeConfig {
    fn normalize(&mut self) {
        self.retained_prefix_tokens = self.retained_prefix_tokens.max(0);
        self.snapshot_interval_tokens = self.snapshot_interval_tokens.max(0);
        self.max_snapshot_entries = self.max_snapshot_entries.max(1);
        self.max_snapshot_bytes = self.max_snapshot_bytes.max(1);
        self.max_session_entries = self.max_session_entries.max(1);
        self.checkpoint_every_tokens = self.checkpoint_every_tokens.max(0);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvReuseMode {
    Disabled,
    LiveSlotPrefix,
    StateSnapshot,
    LiveSlotAndSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheKeyPolicy {
    ContextKey,
    PromptHash,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct MultimodalRuntimeConfig {
    pub projector_path: Option<String>,
    pub use_gpu: Option<bool>,
    pub image_min_tokens: Option<i32>,
    pub image_max_tokens: Option<i32>,
}

impl MultimodalRuntimeConfig {
    fn normalize(&mut self) {
        self.image_min_tokens = positive_or_none(self.image_min_tokens, 0);
        self.image_max_tokens = positive_or_none(self.image_max_tokens, 0);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResidencyRuntimeConfig {
    pub max_gpu_models_per_device: usize,
    pub allow_cpu_models_while_gpu_loaded: bool,
    pub require_gpu_lease: bool,
    pub gpu_memory_safety_margin_bytes: u64,
}

impl Default for ResidencyRuntimeConfig {
    fn default() -> Self {
        Self {
            max_gpu_models_per_device: 1,
            allow_cpu_models_while_gpu_loaded: true,
            require_gpu_lease: true,
            gpu_memory_safety_margin_bytes: 512 * 1024 * 1024,
        }
    }
}

impl ResidencyRuntimeConfig {
    fn normalize(&mut self) {
        self.max_gpu_models_per_device = self.max_gpu_models_per_device.max(1);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ObservabilityRuntimeConfig {
    pub runtime_metrics: bool,
    pub backend_profiling: bool,
}

impl ObservabilityRuntimeConfig {
    pub fn effective_runtime_metrics(self) -> bool {
        self.runtime_metrics || self.backend_profiling
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateOptions {
    pub max_tokens: i32,
    pub stream: bool,
    pub stop: Vec<String>,
    pub sampling: Option<SamplingRuntimeConfig>,
    pub grammar: Option<String>,
    pub json_schema: Option<String>,
    pub cache_key: Option<String>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            stream: false,
            stop: Vec::new(),
            sampling: None,
            grammar: None,
            json_schema: None,
            cache_key: None,
        }
    }
}

pub(super) fn push_arg(args: &mut Vec<String>, key: impl Into<String>, value: impl Into<String>) {
    args.push(key.into());
    args.push(value.into());
}

pub(super) fn bool_arg(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

pub(super) fn join_csv<T>(values: impl IntoIterator<Item = T>) -> String
where
    T: Display,
{
    let mut out = String::new();
    for value in values {
        if !out.is_empty() {
            out.push(',');
        }
        let _ = write!(&mut out, "{value}");
    }
    out
}

pub(super) fn positive_or_none(value: Option<i32>, minimum: i32) -> Option<i32> {
    value.map(|value| value.max(minimum))
}

#[cfg(test)]
mod tests {
    mod inference_config_tests;
}
