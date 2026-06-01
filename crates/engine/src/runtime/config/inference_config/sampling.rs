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

#[cfg(test)]
mod tests {
    mod sampling_tests;
}
