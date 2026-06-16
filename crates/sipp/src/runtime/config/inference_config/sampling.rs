use serde::{Deserialize, Serialize};

/// Request-level sampling override.
#[derive(Debug, Clone, PartialEq)]
pub enum RequestSampling {
    /// Complete local sampler override.
    Full(SamplingRuntimeConfig),
    /// Sparse override for common text-generation knobs.
    Patch(SamplingRuntimePatch),
}

impl From<SamplingRuntimeConfig> for RequestSampling {
    fn from(config: SamplingRuntimeConfig) -> Self {
        Self::Full(config)
    }
}

/// Sparse request-level sampling patch that preserves runtime defaults.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingRuntimePatch {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
}

impl SamplingRuntimePatch {
    /// Returns true when the patch contains no request-level overrides.
    pub fn is_empty(&self) -> bool {
        self.temperature.is_none()
            && self.top_p.is_none()
            && self.repeat_last_n.is_none()
            && self.repeat_penalty.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
    }

    pub(crate) fn apply_to(&self, sampling: &mut SamplingRuntimeConfig) {
        if let Some(value) = self.temperature {
            sampling.temperature = Some(value);
        }
        if let Some(value) = self.top_p {
            sampling.top_p = Some(value);
        }
        if let Some(value) = self.repeat_last_n {
            sampling.repeat_last_n = Some(value);
        }
        if let Some(value) = self.repeat_penalty {
            sampling.repeat_penalty = Some(value);
        }
        if let Some(value) = self.frequency_penalty {
            sampling.frequency_penalty = Some(value);
        }
        if let Some(value) = self.presence_penalty {
            sampling.presence_penalty = Some(value);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
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

const DEFAULT_SAMPLERS: [SamplerStage; 4] = [
    SamplerStage::TopK,
    SamplerStage::Penalties,
    SamplerStage::TopP,
    SamplerStage::Temperature,
];

impl Default for SamplingRuntimeConfig {
    fn default() -> Self {
        Self {
            samplers: DEFAULT_SAMPLERS.to_vec(),
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

impl SamplingRuntimeConfig {
    pub(crate) fn prompt_sampler_seed_start(&self, prompt_len: usize) -> usize {
        let Some(history_len) = self.finite_prompt_history_len() else {
            return 0;
        };
        prompt_len.saturating_sub(history_len)
    }

    fn finite_prompt_history_len(&self) -> Option<usize> {
        if self.mirostat.unwrap_or(0) != 0 {
            return Some(0);
        }

        let mut history_len = 0;
        if self.stage_enabled(SamplerStage::Penalties) && self.penalties_enabled() {
            update_history_len(
                &mut history_len,
                self.repeat_last_n.unwrap_or(CPP_DEFAULT_REPEAT_LAST_N),
            )?;
        }
        if self.stage_enabled(SamplerStage::Dry) && self.dry_enabled() {
            update_history_len(
                &mut history_len,
                self.dry_penalty_last_n
                    .unwrap_or(CPP_DEFAULT_DRY_PENALTY_LAST_N),
            )?;
        }
        Some(history_len)
    }

    fn stage_enabled(&self, stage: SamplerStage) -> bool {
        self.samplers.is_empty() && matches!(stage, SamplerStage::Penalties | SamplerStage::Dry)
            || self.samplers.contains(&stage)
    }

    fn penalties_enabled(&self) -> bool {
        self.repeat_last_n.unwrap_or(CPP_DEFAULT_REPEAT_LAST_N) != 0
            && (self.repeat_penalty.unwrap_or(CPP_DEFAULT_REPEAT_PENALTY)
                != CPP_DEFAULT_REPEAT_PENALTY
                || self
                    .frequency_penalty
                    .unwrap_or(CPP_DEFAULT_FREQUENCY_PENALTY)
                    != CPP_DEFAULT_FREQUENCY_PENALTY
                || self
                    .presence_penalty
                    .unwrap_or(CPP_DEFAULT_PRESENCE_PENALTY)
                    != CPP_DEFAULT_PRESENCE_PENALTY)
    }

    fn dry_enabled(&self) -> bool {
        self.dry_multiplier.unwrap_or(CPP_DEFAULT_DRY_MULTIPLIER) != 0.0
            && self.dry_base.unwrap_or(CPP_DEFAULT_DRY_BASE) >= 1.0
            && self
                .dry_penalty_last_n
                .unwrap_or(CPP_DEFAULT_DRY_PENALTY_LAST_N)
                != 0
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

pub(super) fn merge_sampling_override_json(
    base: &mut serde_json::Value,
    override_value: serde_json::Value,
) {
    let (Some(base), Some(override_map)) = (base.as_object_mut(), override_value.as_object())
    else {
        return;
    };

    for (key, value) in override_map {
        if should_merge_sampling_override(value) {
            base.insert(key.clone(), value.clone());
        }
    }
}

fn should_merge_sampling_override(value: &serde_json::Value) -> bool {
    match value {
        serde_json::Value::Null => false,
        serde_json::Value::Array(items) => !items.is_empty(),
        _ => true,
    }
}

fn update_history_len(history_len: &mut usize, last_n: i32) -> Option<()> {
    if last_n < 0 {
        return None;
    }
    *history_len = (*history_len).max(last_n as usize);
    Some(())
}

const CPP_DEFAULT_REPEAT_LAST_N: i32 = 64;
const CPP_DEFAULT_DRY_PENALTY_LAST_N: i32 = -1;
const CPP_DEFAULT_REPEAT_PENALTY: f32 = 1.0;
const CPP_DEFAULT_FREQUENCY_PENALTY: f32 = 0.0;
const CPP_DEFAULT_PRESENCE_PENALTY: f32 = 0.0;
const CPP_DEFAULT_DRY_MULTIPLIER: f32 = 0.0;
const CPP_DEFAULT_DRY_BASE: f32 = 1.75;

#[cfg(test)]
#[path = "../../../tests/runtime/config/inference_config/sampling_tests.rs"]
mod sampling_tests;
