//! NativeRuntimeConfig and friends: typed runtime configuration plus serialization to the shim JSON shape.

use std::fmt::{Display, Write as _};

use serde::{Deserialize, Serialize};

use crate::defaults::{BYTES_PER_MIB, BYTES_PER_MIB_U64};
use crate::runtime::numeric::{nonnegative_i32, positive_i32, positive_usize};

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
pub use sampling::{RequestSampling, SamplingRuntimePatch};

pub const DEFAULT_CONTEXT_KEY: &str = "default";
pub const DEFAULT_MAX_TOKENS: i32 = 64;
pub(super) const KEY_VALUE_ARG_LEN: usize = 2;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
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
        self.placement.normalize();
        self.context.normalize();
        self.scheduler.normalize();
        self.cache.normalize();
        self.multimodal.normalize();
        self.residency.normalize();
        self
    }

    pub fn llama_common_args(&self) -> Vec<String> {
        let normalized = self.clone().normalize();
        let mut args =
            Vec::with_capacity(normalized.placement.arg_len() + normalized.context.arg_len());
        normalized.placement.push_args(&mut args);
        normalized.context.push_args(&mut args);
        args
    }

    pub fn try_sampling_json_with_override(
        &self,
        override_config: Option<&RequestSampling>,
    ) -> serde_json::Result<String> {
        match override_config {
            None => serde_json::to_string(&self.sampling),
            Some(RequestSampling::Patch(patch)) => {
                let mut sampling = self.sampling.clone();
                patch.apply_to(&mut sampling);
                serde_json::to_string(&sampling)
            }
            Some(RequestSampling::Full(config)) => {
                let mut value = serde_json::to_value(&self.sampling)?;
                let override_value = serde_json::to_value(config)?;
                merge_sampling_override_json(&mut value, override_value);
                serde_json::to_string(&value)
            }
        }
    }

    pub(crate) fn prompt_sampler_seed_start(
        &self,
        override_config: Option<&RequestSampling>,
        prompt_len: usize,
    ) -> usize {
        let mut patched_sampling;
        let sampling = match override_config {
            Some(RequestSampling::Full(config)) => config,
            Some(RequestSampling::Patch(patch)) => {
                patched_sampling = self.sampling.clone();
                patch.apply_to(&mut patched_sampling);
                &patched_sampling
            }
            None => &self.sampling,
        };
        sampling.prompt_sampler_seed_start(prompt_len)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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
        self.policy.decode_token_reserve = nonnegative_i32(self.policy.decode_token_reserve);
        self.prefill_chunk_size = nonnegative_i32(self.prefill_chunk_size);
        self.max_running_requests = positive_or_none(self.max_running_requests, 1);
        self.max_queued_requests = positive_or_none(self.max_queued_requests, 1);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CacheRuntimeConfig {
    pub mode: KvReuseMode,
    pub retained_prefix_tokens: i32,
    pub snapshot_interval_tokens: i32,
    pub max_snapshot_entries: i32,
    pub max_snapshot_bytes: usize,
}

impl Default for CacheRuntimeConfig {
    fn default() -> Self {
        Self {
            mode: KvReuseMode::LiveSlotPrefix,
            retained_prefix_tokens: 100,
            snapshot_interval_tokens: 128,
            max_snapshot_entries: 32,
            max_snapshot_bytes: 256 * BYTES_PER_MIB,
        }
    }
}

impl CacheRuntimeConfig {
    fn normalize(&mut self) {
        self.retained_prefix_tokens = nonnegative_i32(self.retained_prefix_tokens);
        self.snapshot_interval_tokens = nonnegative_i32(self.snapshot_interval_tokens);
        self.max_snapshot_entries = positive_i32(self.max_snapshot_entries);
        self.max_snapshot_bytes = positive_usize(self.max_snapshot_bytes);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum KvReuseMode {
    Disabled,
    #[default]
    LiveSlotPrefix,
    StateSnapshot,
    LiveSlotAndSnapshot,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
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
#[serde(default, deny_unknown_fields)]
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
            gpu_memory_safety_margin_bytes: 512 * BYTES_PER_MIB_U64,
        }
    }
}

impl ResidencyRuntimeConfig {
    fn normalize(&mut self) {
        self.max_gpu_models_per_device = positive_usize(self.max_gpu_models_per_device);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
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
            max_tokens: DEFAULT_MAX_TOKENS,
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

pub(super) fn flag_len(enabled: bool) -> usize {
    usize::from(enabled)
}

pub(super) fn key_value_args_len(enabled: impl IntoIterator<Item = bool>) -> usize {
    enabled
        .into_iter()
        .map(|enabled| if enabled { KEY_VALUE_ARG_LEN } else { 0 })
        .sum()
}

pub(super) fn flags_len(enabled: impl IntoIterator<Item = bool>) -> usize {
    enabled.into_iter().map(flag_len).sum()
}

pub(super) fn args_len(
    base_len: usize,
    key_value_args: impl IntoIterator<Item = bool>,
    flags: impl IntoIterator<Item = bool>,
) -> usize {
    base_len + key_value_args_len(key_value_args) + flags_len(flags)
}

pub(super) fn push_optional_arg<T: Display>(args: &mut Vec<String>, key: &str, value: Option<T>) {
    if let Some(value) = value {
        push_arg(args, key, value.to_string());
    }
}

pub(super) fn push_csv_arg<T>(
    args: &mut Vec<String>,
    key: &str,
    values: impl IntoIterator<Item = T>,
) where
    T: Display,
{
    push_arg(args, key, join_csv(values));
}

pub(super) fn push_flag(args: &mut Vec<String>, flag: &str, enabled: bool) {
    if enabled {
        args.push(flag.to_string());
    }
}

pub(super) fn push_flag_pair(
    args: &mut Vec<String>,
    enabled: bool,
    enabled_flag: &str,
    disabled_flag: &str,
) {
    args.push(if enabled { enabled_flag } else { disabled_flag }.to_string());
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

pub(super) fn positive_or_default(value: Option<i32>, default: i32, minimum: i32) -> i32 {
    value.unwrap_or(default).max(minimum)
}

#[cfg(test)]
pub(super) fn arg_value<'args>(args: &'args [String], key: &str) -> Option<&'args str> {
    args.windows(2)
        .find_map(|window| (window[0] == key).then_some(window[1].as_str()))
}

#[cfg(test)]
#[path = "../../../tests/runtime/config/inference_config_tests.rs"]
mod inference_config_tests;
