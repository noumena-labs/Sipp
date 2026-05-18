use serde::{Deserialize, Serialize};

use super::SchedulerPolicyConfig;
#[cfg(test)]
use super::SchedulerPolicyMode;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

impl Default for NativeRuntimeConfig {
    fn default() -> Self {
        Self {
            placement: ModelPlacementConfig::default(),
            context: ContextRuntimeConfig::default(),
            sampling: SamplingRuntimeConfig::default(),
            scheduler: SchedulerRuntimeConfig::default(),
            cache: CacheRuntimeConfig::default(),
            multimodal: MultimodalRuntimeConfig::default(),
            residency: ResidencyRuntimeConfig::default(),
            observability: ObservabilityRuntimeConfig::default(),
        }
    }
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
        let mut args = Vec::new();
        self.placement.push_args(&mut args);
        self.context.push_args(&mut args);
        args
    }

    pub fn sampling_json(&self) -> String {
        self.sampling_json_with_override(None)
    }

    pub fn sampling_json_with_override(
        &self,
        override_config: Option<&SamplingRuntimeConfig>,
    ) -> String {
        let mut value =
            serde_json::to_value(&self.sampling).unwrap_or_else(|_| serde_json::json!({}));
        if let Some(override_config) = override_config {
            if let Ok(override_value) = serde_json::to_value(override_config) {
                merge_sampling_override_json(&mut value, override_value);
            }
        }
        serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelPlacementConfig {
    pub devices: Vec<String>,
    pub gpu_layers: GpuLayerConfig,
    pub split_mode: SplitMode,
    pub main_gpu: Option<i32>,
    pub tensor_split: Vec<f32>,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub fit_params: bool,
    pub fit_params_min_ctx: Option<i32>,
    pub fit_params_target_bytes: Vec<u64>,
    pub check_tensors: bool,
    pub no_extra_bufts: bool,
    pub no_host: bool,
}

impl Default for ModelPlacementConfig {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            gpu_layers: GpuLayerConfig::Auto,
            split_mode: SplitMode::Layer,
            main_gpu: None,
            tensor_split: Vec::new(),
            use_mmap: cfg!(not(target_family = "wasm")),
            use_mlock: false,
            fit_params: false,
            fit_params_min_ctx: None,
            fit_params_target_bytes: Vec::new(),
            check_tensors: false,
            no_extra_bufts: false,
            no_host: false,
        }
    }
}

impl ModelPlacementConfig {
    fn push_args(&self, args: &mut Vec<String>) {
        if !self.devices.is_empty() {
            push_arg(args, "--device", self.devices.join(","));
        }
        match self.gpu_layers {
            GpuLayerConfig::Auto => push_arg(args, "--gpu-layers", "auto"),
            GpuLayerConfig::All => push_arg(args, "--gpu-layers", "all"),
            GpuLayerConfig::Count(count) => push_arg(args, "--gpu-layers", count.to_string()),
        }
        push_arg(args, "--split-mode", self.split_mode.as_llama_arg());
        if let Some(main_gpu) = self.main_gpu {
            push_arg(args, "--main-gpu", main_gpu.to_string());
        }
        if !self.tensor_split.is_empty() {
            push_arg(
                args,
                "--tensor-split",
                self.tensor_split
                    .iter()
                    .map(|value| value.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        push_arg(args, "--fit", bool_arg(self.fit_params));
        if let Some(min_ctx) = self.fit_params_min_ctx {
            push_arg(args, "--fit-ctx", min_ctx.to_string());
        }
        if !self.fit_params_target_bytes.is_empty() {
            push_arg(
                args,
                "--fit-target",
                self.fit_params_target_bytes
                    .iter()
                    .map(|bytes| (bytes / (1024 * 1024)).to_string())
                    .collect::<Vec<_>>()
                    .join(","),
            );
        }
        if self.use_mlock {
            args.push("--mlock".to_string());
        }
        if !self.use_mmap {
            args.push("--no-mmap".to_string());
        }
        if self.check_tensors {
            args.push("--check-tensors".to_string());
        }
        if self.no_extra_bufts {
            args.push("--no-repack".to_string());
        }
        if self.no_host {
            args.push("--no-host".to_string());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuLayerConfig {
    Auto,
    All,
    Count(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitMode {
    None,
    Layer,
    Row,
    Tensor,
}

impl SplitMode {
    fn as_llama_arg(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Layer => "layer",
            Self::Row => "row",
            Self::Tensor => "tensor",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextRuntimeConfig {
    pub n_ctx: Option<i32>,
    pub n_batch: Option<i32>,
    pub n_ubatch: Option<i32>,
    pub n_parallel: Option<i32>,
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    pub flash_attention: FlashAttentionMode,
    pub kv_unified: Option<bool>,
    pub cache_type_k: KvCacheType,
    pub cache_type_v: KvCacheType,
    pub offload_kqv: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub warmup: bool,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    pub yarn_orig_ctx: Option<i32>,
    pub yarn_ext_factor: Option<f32>,
    pub yarn_attn_factor: Option<f32>,
    pub yarn_beta_fast: Option<f32>,
    pub yarn_beta_slow: Option<f32>,
}

impl Default for ContextRuntimeConfig {
    fn default() -> Self {
        Self {
            n_ctx: None,
            n_batch: None,
            n_ubatch: None,
            n_parallel: Some(1),
            n_threads: None,
            n_threads_batch: None,
            flash_attention: FlashAttentionMode::Auto,
            kv_unified: None,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            offload_kqv: true,
            op_offload: true,
            swa_full: false,
            warmup: true,
            rope_scaling: None,
            rope_freq_base: None,
            rope_freq_scale: None,
            yarn_orig_ctx: None,
            yarn_ext_factor: None,
            yarn_attn_factor: None,
            yarn_beta_fast: None,
            yarn_beta_slow: None,
        }
    }
}

impl ContextRuntimeConfig {
    fn normalize(&mut self) {
        self.n_ctx = positive_or_none(self.n_ctx, 1);
        self.n_batch = positive_or_none(self.n_batch, 1);
        self.n_ubatch = positive_or_none(self.n_ubatch, 1);
        self.n_parallel = Some(self.n_parallel.unwrap_or(1).max(1));
        self.n_threads = positive_or_none(self.n_threads, 0);
        self.n_threads_batch = positive_or_none(self.n_threads_batch, 0);
    }

    fn push_args(&self, args: &mut Vec<String>) {
        if let Some(value) = self.n_ctx {
            push_arg(args, "--ctx-size", value.to_string());
        }
        if let Some(value) = self.n_batch {
            push_arg(args, "--batch-size", value.to_string());
        }
        if let Some(value) = self.n_ubatch {
            push_arg(args, "--ubatch-size", value.to_string());
        }
        if let Some(value) = self.n_parallel {
            push_arg(args, "--parallel", value.to_string());
        }
        if let Some(value) = self.n_threads {
            push_arg(args, "--threads", value.to_string());
        }
        if let Some(value) = self.n_threads_batch {
            push_arg(args, "--threads-batch", value.to_string());
        }
        push_arg(args, "--flash-attn", self.flash_attention.as_llama_arg());
        if let Some(value) = self.kv_unified {
            args.push(
                if value {
                    "--kv-unified"
                } else {
                    "--no-kv-unified"
                }
                .to_string(),
            );
        }
        push_arg(args, "--cache-type-k", self.cache_type_k.as_llama_arg());
        push_arg(args, "--cache-type-v", self.cache_type_v.as_llama_arg());
        args.push(
            if self.offload_kqv {
                "--kv-offload"
            } else {
                "--no-kv-offload"
            }
            .to_string(),
        );
        args.push(
            if self.op_offload {
                "--op-offload"
            } else {
                "--no-op-offload"
            }
            .to_string(),
        );
        if self.swa_full {
            args.push("--swa-full".to_string());
        }
        args.push(
            if self.warmup {
                "--warmup"
            } else {
                "--no-warmup"
            }
            .to_string(),
        );
        if let Some(value) = self.rope_scaling {
            push_arg(args, "--rope-scaling", value.as_llama_arg());
        }
        if let Some(value) = self.rope_freq_base {
            push_arg(args, "--rope-freq-base", value.to_string());
        }
        if let Some(value) = self.rope_freq_scale {
            push_arg(args, "--rope-freq-scale", value.to_string());
        }
        if let Some(value) = self.yarn_orig_ctx {
            push_arg(args, "--yarn-orig-ctx", value.to_string());
        }
        if let Some(value) = self.yarn_ext_factor {
            push_arg(args, "--yarn-ext-factor", value.to_string());
        }
        if let Some(value) = self.yarn_attn_factor {
            push_arg(args, "--yarn-attn-factor", value.to_string());
        }
        if let Some(value) = self.yarn_beta_fast {
            push_arg(args, "--yarn-beta-fast", value.to_string());
        }
        if let Some(value) = self.yarn_beta_slow {
            push_arg(args, "--yarn-beta-slow", value.to_string());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttentionMode {
    Auto,
    Enabled,
    Disabled,
}

impl Default for FlashAttentionMode {
    fn default() -> Self {
        Self::Auto
    }
}

impl FlashAttentionMode {
    fn as_llama_arg(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Enabled => "on",
            Self::Disabled => "off",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheType {
    F16,
    F32,
    Q8_0,
    Q4_0,
    Q4_1,
    Iq4Nl,
    Q5_0,
    Q5_1,
}

impl Default for KvCacheType {
    fn default() -> Self {
        Self::F16
    }
}

impl KvCacheType {
    fn as_llama_arg(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Iq4Nl => "iq4_nl",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeScaling {
    None,
    Linear,
    Yarn,
}

impl RopeScaling {
    fn as_llama_arg(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Yarn => "yarn",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingRuntimeConfig {
    pub samplers: Vec<SamplerStage>,
    pub seed: Option<u32>,
    pub top_k: Option<i32>,
    pub top_p: Option<f32>,
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub xtc_probability: Option<f32>,
    pub xtc_threshold: Option<f32>,
    pub top_n_sigma: Option<f32>,
    pub temperature: Option<f32>,
    pub dynatemp_range: Option<f32>,
    pub dynatemp_exponent: Option<f32>,
    pub repeat_last_n: Option<i32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<i32>,
    pub dry_penalty_last_n: Option<i32>,
    pub dry_sequence_breakers: Vec<String>,
    pub mirostat: Option<i32>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub min_keep: Option<i32>,
    pub n_probs: Option<i32>,
    pub logit_bias: Vec<LogitBias>,
    pub ignore_eos: bool,
    pub grammar_lazy: bool,
    pub preserved_tokens: Vec<i32>,
    pub backend_sampling: bool,
}

impl Default for SamplingRuntimeConfig {
    fn default() -> Self {
        Self {
            samplers: vec![
                SamplerStage::TopK,
                SamplerStage::Penalties,
                SamplerStage::TopP,
                SamplerStage::Temperature,
            ],
            seed: None,
            top_k: Some(40),
            top_p: Some(0.8),
            min_p: None,
            typical_p: None,
            xtc_probability: None,
            xtc_threshold: None,
            top_n_sigma: None,
            temperature: Some(0.7),
            dynatemp_range: None,
            dynatemp_exponent: None,
            repeat_last_n: Some(64),
            repeat_penalty: Some(1.05),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            dry_multiplier: None,
            dry_base: None,
            dry_allowed_length: None,
            dry_penalty_last_n: None,
            dry_sequence_breakers: Vec::new(),
            mirostat: None,
            mirostat_tau: None,
            mirostat_eta: None,
            min_keep: None,
            n_probs: None,
            logit_bias: Vec::new(),
            ignore_eos: false,
            grammar_lazy: false,
            preserved_tokens: Vec::new(),
            backend_sampling: cfg!(not(target_arch = "wasm32")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplerStage {
    Dry,
    TopK,
    TypicalP,
    TopP,
    TopNSigma,
    MinP,
    Xtc,
    Temperature,
    Infill,
    Penalties,
    AdaptiveP,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LogitBias {
    pub token: i32,
    pub bias: f32,
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

fn push_arg(args: &mut Vec<String>, key: impl Into<String>, value: impl Into<String>) {
    args.push(key.into());
    args.push(value.into());
}

fn bool_arg(value: bool) -> &'static str {
    if value {
        "on"
    } else {
        "off"
    }
}

fn positive_or_none(value: Option<i32>, minimum: i32) -> Option<i32> {
    value.map(|value| value.max(minimum))
}

fn merge_sampling_override_json(base: &mut serde_json::Value, override_value: serde_json::Value) {
    let (Some(base), Some(override_map)) = (base.as_object_mut(), override_value.as_object())
    else {
        return;
    };

    for (key, value) in override_map {
        let should_merge = match value {
            serde_json::Value::Null => false,
            serde_json::Value::Array(items) => !items.is_empty(),
            _ => true,
        };
        if should_merge {
            base.insert(key.clone(), value.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_defaults_match_legacy_cpp_browser_runtime() {
        let sampling = SamplingRuntimeConfig::default();

        assert_eq!(
            sampling.samplers,
            vec![
                SamplerStage::TopK,
                SamplerStage::Penalties,
                SamplerStage::TopP,
                SamplerStage::Temperature,
            ]
        );
        assert_eq!(sampling.top_k, Some(40));
        assert_eq!(sampling.top_p, Some(0.8));
        assert_eq!(sampling.temperature, Some(0.7));
        assert_eq!(sampling.repeat_last_n, Some(64));
        assert_eq!(sampling.repeat_penalty, Some(1.05));
        assert_eq!(sampling.frequency_penalty, Some(0.0));
        assert_eq!(sampling.presence_penalty, Some(0.0));
        assert_eq!(sampling.backend_sampling, cfg!(not(target_arch = "wasm32")));
    }

    #[test]
    fn native_runtime_config_deserializes_sparse_browser_json() {
        let config: NativeRuntimeConfig = serde_json::from_str(
            r#"{
                "placement": { "gpu_layers": { "count": 99 } },
                "context": { "n_ctx": 8192, "flash_attention": "enabled" },
                "sampling": {
                    "samplers": ["top_k", "top_p", "temperature"],
                    "typical_p": 0.95,
                    "backend_sampling": true
                },
                "scheduler": {
                    "policy": {
                        "mode": "throughput_first",
                        "decode_token_reserve": 2
                    }
                }
            }"#,
        )
        .expect("browser runtime json");

        assert_eq!(config.placement.gpu_layers, GpuLayerConfig::Count(99));
        assert_eq!(config.context.n_ctx, Some(8192));
        assert_eq!(config.context.flash_attention, FlashAttentionMode::Enabled);
        assert_eq!(
            config.sampling.samplers,
            vec![
                SamplerStage::TopK,
                SamplerStage::TopP,
                SamplerStage::Temperature
            ]
        );
        assert_eq!(config.sampling.typical_p, Some(0.95));
        assert!(config.sampling.backend_sampling);
        assert_eq!(
            config.scheduler.policy.mode,
            SchedulerPolicyMode::ThroughputFirst
        );
        assert_eq!(config.scheduler.policy.decode_token_reserve, 2);
        assert!(!config.scheduler.policy.enable_adaptive_prefill_chunking);
    }
}
