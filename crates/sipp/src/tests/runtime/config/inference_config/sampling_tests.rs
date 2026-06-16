//! Tests the `runtime::config::inference_config::sampling` module in `sipp`.
//!
//! Covers runtime configuration normalization, serialization, and boundary choices through pure value assertions.

use super::{SamplerStage, SamplingRuntimeConfig, SamplingRuntimeOverride};

#[test]
fn sampling_override_reports_empty_only_when_no_fields_are_set() {
    assert!(SamplingRuntimeOverride::default().is_empty());
    assert!(!SamplingRuntimeOverride {
        temperature: Some(0.1),
        ..SamplingRuntimeOverride::default()
    }
    .is_empty());
    assert!(!SamplingRuntimeOverride {
        samplers: Some(Vec::new()),
        ..SamplingRuntimeOverride::default()
    }
    .is_empty());
    assert!(!SamplingRuntimeOverride {
        backend_sampling: Some(false),
        ..SamplingRuntimeOverride::default()
    }
    .is_empty());
}

#[test]
fn sampling_override_applies_scalars_and_explicit_arrays() {
    let mut sampling = SamplingRuntimeConfig::default();
    let override_config = SamplingRuntimeOverride {
        samplers: Some(vec![SamplerStage::TopK]),
        temperature: Some(0.2),
        repeat_last_n: Some(128),
        dry_sequence_breakers: Some(Vec::new()),
        backend_sampling: Some(false),
        ..SamplingRuntimeOverride::default()
    };

    override_config.apply_to(&mut sampling);

    assert_eq!(sampling.samplers, vec![SamplerStage::TopK]);
    assert_eq!(sampling.temperature, Some(0.2));
    assert_eq!(sampling.repeat_last_n, Some(128));
    assert!(sampling.dry_sequence_breakers.is_empty());
    assert!(!sampling.backend_sampling);
}
