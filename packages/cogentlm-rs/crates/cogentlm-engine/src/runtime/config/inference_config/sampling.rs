use serde::{Deserialize, Serialize};

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

pub(super) fn merge_sampling_override_json(
    base: &mut serde_json::Value,
    override_value: serde_json::Value,
) {
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
    use super::{SamplerStage, SamplingRuntimeConfig};

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
    fn sampling_override_ignores_nulls_and_empty_arrays() {
        let mut base = serde_json::json!({
            "top_k": 40,
            "samplers": ["top_k"],
            "backend_sampling": true
        });
        let override_value = serde_json::json!({
            "top_k": 12,
            "samplers": [],
            "backend_sampling": null
        });

        super::merge_sampling_override_json(&mut base, override_value);

        assert_eq!(base["top_k"], 12);
        assert_eq!(base["samplers"], serde_json::json!(["top_k"]));
        assert_eq!(base["backend_sampling"], true);
    }
}
