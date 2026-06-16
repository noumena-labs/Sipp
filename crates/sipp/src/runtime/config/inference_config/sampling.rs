use serde::{Deserialize, Serialize};

/// Request-level sampling override applied over runtime defaults.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SamplingRuntimeOverride {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub samplers: Option<Vec<SamplerStage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub typical_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xtc_probability: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub xtc_threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n_sigma: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynatemp_range: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dynatemp_exponent: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_multiplier: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_base: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_allowed_length: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_penalty_last_n: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dry_sequence_breakers: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_tau: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mirostat_eta: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_keep: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_probs: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<Vec<LogitBias>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore_eos: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grammar_lazy: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preserved_tokens: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub backend_sampling: Option<bool>,
}

impl SamplingRuntimeOverride {
    /// Returns true when no request-level override fields are set.
    pub fn is_empty(&self) -> bool {
        self.samplers.is_none()
            && self.seed.is_none()
            && self.top_k.is_none()
            && self.temperature.is_none()
            && self.top_p.is_none()
            && self.min_p.is_none()
            && self.typical_p.is_none()
            && self.xtc_probability.is_none()
            && self.xtc_threshold.is_none()
            && self.top_n_sigma.is_none()
            && self.dynatemp_range.is_none()
            && self.dynatemp_exponent.is_none()
            && self.repeat_last_n.is_none()
            && self.repeat_penalty.is_none()
            && self.frequency_penalty.is_none()
            && self.presence_penalty.is_none()
            && self.dry_multiplier.is_none()
            && self.dry_base.is_none()
            && self.dry_allowed_length.is_none()
            && self.dry_penalty_last_n.is_none()
            && self.dry_sequence_breakers.is_none()
            && self.mirostat.is_none()
            && self.mirostat_tau.is_none()
            && self.mirostat_eta.is_none()
            && self.min_keep.is_none()
            && self.n_probs.is_none()
            && self.logit_bias.is_none()
            && self.ignore_eos.is_none()
            && self.grammar_lazy.is_none()
            && self.preserved_tokens.is_none()
            && self.backend_sampling.is_none()
    }

    /// Applies set override fields to a full runtime sampling config.
    pub fn apply_to(&self, sampling: &mut SamplingRuntimeConfig) {
        if let Some(value) = &self.samplers {
            sampling.samplers = value.clone();
        }
        if let Some(value) = self.seed {
            sampling.seed = Some(value);
        }
        if let Some(value) = self.top_k {
            sampling.top_k = Some(value);
        }
        if let Some(value) = self.temperature {
            sampling.temperature = Some(value);
        }
        if let Some(value) = self.top_p {
            sampling.top_p = Some(value);
        }
        if let Some(value) = self.min_p {
            sampling.min_p = Some(value);
        }
        if let Some(value) = self.typical_p {
            sampling.typical_p = Some(value);
        }
        if let Some(value) = self.xtc_probability {
            sampling.xtc_probability = Some(value);
        }
        if let Some(value) = self.xtc_threshold {
            sampling.xtc_threshold = Some(value);
        }
        if let Some(value) = self.top_n_sigma {
            sampling.top_n_sigma = Some(value);
        }
        if let Some(value) = self.dynatemp_range {
            sampling.dynatemp_range = Some(value);
        }
        if let Some(value) = self.dynatemp_exponent {
            sampling.dynatemp_exponent = Some(value);
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
        if let Some(value) = self.dry_multiplier {
            sampling.dry_multiplier = Some(value);
        }
        if let Some(value) = self.dry_base {
            sampling.dry_base = Some(value);
        }
        if let Some(value) = self.dry_allowed_length {
            sampling.dry_allowed_length = Some(value);
        }
        if let Some(value) = self.dry_penalty_last_n {
            sampling.dry_penalty_last_n = Some(value);
        }
        if let Some(value) = &self.dry_sequence_breakers {
            sampling.dry_sequence_breakers = value.clone();
        }
        if let Some(value) = self.mirostat {
            sampling.mirostat = Some(value);
        }
        if let Some(value) = self.mirostat_tau {
            sampling.mirostat_tau = Some(value);
        }
        if let Some(value) = self.mirostat_eta {
            sampling.mirostat_eta = Some(value);
        }
        if let Some(value) = self.min_keep {
            sampling.min_keep = Some(value);
        }
        if let Some(value) = self.n_probs {
            sampling.n_probs = Some(value);
        }
        if let Some(value) = &self.logit_bias {
            sampling.logit_bias = value.clone();
        }
        if let Some(value) = self.ignore_eos {
            sampling.ignore_eos = value;
        }
        if let Some(value) = self.grammar_lazy {
            sampling.grammar_lazy = value;
        }
        if let Some(value) = &self.preserved_tokens {
            sampling.preserved_tokens = value.clone();
        }
        if let Some(value) = self.backend_sampling {
            sampling.backend_sampling = value;
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
