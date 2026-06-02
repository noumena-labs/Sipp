//! Tests the `map` module in `cogentlm-client`.
//!
//! Covers endpoint resolution, remote configuration, facade validation, and run wrappers with deterministic fakes rather than a live local engine.

use cogentlm_engine::engine::SamplingRuntimeConfig;

use super::*;
use crate::{CogentQueryRequest, LocalTextOptions};

#[test]
fn common_sampling_without_local_sampler_builds_sparse_patch() {
    let request = CogentQueryRequest {
        prompt: "hello".to_string(),
        options: CogentTextOptions {
            temperature: Some(0.2),
            ..CogentTextOptions::default()
        },
        ..CogentQueryRequest::default()
    };

    let request = local_query_request(request).expect("local query request");

    let Some(RequestSampling::Patch(patch)) = request.options.sampling else {
        panic!("common-only sampling should use sparse patch");
    };
    assert_eq!(patch.temperature, Some(0.2));
    assert_eq!(patch.top_p, None);
}

#[test]
fn common_sampling_conflicts_with_different_explicit_local_sampler() {
    let request = CogentQueryRequest {
        prompt: "hello".to_string(),
        options: CogentTextOptions {
            temperature: Some(0.2),
            ..CogentTextOptions::default()
        },
        local: LocalTextOptions {
            sampling: Some(SamplingRuntimeConfig {
                temperature: Some(0.7),
                ..SamplingRuntimeConfig::default()
            }),
            ..LocalTextOptions::default()
        },
        ..CogentQueryRequest::default()
    };

    let error = match local_query_request(request) {
        Err(error) => error,
        Ok(_) => panic!("conflict must reject"),
    };

    assert!(
        matches!(error, CogentError::InvalidRequest(message) if message.contains("temperature"))
    );
}

#[test]
fn usage_from_stats_omits_negative_counts_and_unchecked_total() {
    let usage = usage_from_stats(RequestStats {
        input_tokens: -1,
        output_tokens: 3,
        ..RequestStats::default()
    });

    assert_eq!(usage.input_tokens, None);
    assert_eq!(usage.output_tokens, Some(3));
    assert_eq!(usage.total_tokens, None);
}
