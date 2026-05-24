//! Unit tests for the stats translation module.

use std::time::Duration;

use serde_json::json;

use super::super::*;
use crate::backend::{json_strings, KEY_NAME};
use crate::engine::protocol::{FinishReason, PoolingType};
use crate::error::Error;
use crate::runtime::metrics::RuntimeObservabilityMetrics;
use crate::runtime::numeric::duration_millis_u64;
use crate::runtime::request::{GenerateResponse, GenerateResponseStatus, ResponseOutput};

#[test]
fn runtime_metrics_map_to_engine_stats() {
    let stats = engine_stats_from_runtime(RuntimeObservabilityMetrics {
        ttft_ms: 10.0,
        itl_avg_ms: 5.0,
        e2e_ms: 100.0,
        prefill_ms: 25.0,
        decode_ms: 75.0,
        native_gpu_ms: 60.0,
        native_sync_ms: 15.0,
        native_logic_ms: 2.0,
        input_tokens: 8,
        output_tokens: 4,
        cache_hits: 3,
        prefill_tokens: 5,
        ..RuntimeObservabilityMetrics::default()
    });

    assert_eq!(stats.input_tokens, 8);
    assert_eq!(stats.output_tokens, 4);
    assert_eq!(stats.cache_hits, 3);
    assert_eq!(stats.prefill_tokens, 5);
    assert_eq!(stats.ttft_ms, Some(10.0));
    assert_eq!(stats.inter_token_ms, Some(5.0));
    assert_eq!(stats.e2e_ms, Some(100.0));
    assert_eq!(stats.tokens_per_second, Some(40.0));
    assert_eq!(stats.prefill_tokens_per_second, Some(200.0));
    assert_eq!(stats.backend_ms, 60.0);
    assert_eq!(stats.sync_ms, 15.0);
    assert_eq!(stats.engine_overhead_ms, 2.0);
}

#[test]
fn backend_observability_parsers_preserve_array_capacity() {
    let names = json_strings(
        &[
            json!({"name": "cpu"}),
            json!({"missing": true}),
            json!({"name": "cuda"}),
        ],
        KEY_NAME,
    );
    assert_eq!(names, vec!["cpu", "cuda"]);
    assert!(names.capacity() >= 3);

    let devices = parse_backend_devices(&[
        json!({"deviceId": "0", "name": "GPU", "type": "cuda", "memoryTotalBytes": 8}),
        json!({"name": "CPU"}),
    ]);
    assert_eq!(devices.len(), 2);
    assert!(devices.capacity() >= 2);
    assert_eq!(devices[0].id.as_deref(), Some("0"));
    assert_eq!(devices[0].memory_total_bytes, Some(8));
    assert_eq!(devices[1].device_type, "unknown");
}

#[test]
fn completed_response_maps_to_generation_result() {
    let response = GenerateResponse {
        runtime_observability: RuntimeObservabilityMetrics {
            e2e_ms: 50.0,
            output_tokens: 5,
            ..RuntimeObservabilityMetrics::default()
        },
        ..GenerateResponse::completed(7, "hello")
    };
    let result = generation_result_from_response(response).expect("generation result");

    assert_eq!(result.id, "7");
    assert_eq!(result.text, "hello");
    assert_eq!(result.finish_reason, FinishReason::Stop);
    assert_eq!(result.stats.output_tokens, 5);
    assert_eq!(result.stats.tokens_per_second, Some(100.0));
}

#[test]
fn embedding_response_is_not_a_generation_result() {
    let response = GenerateResponse::terminal(
        9,
        GenerateResponseStatus::Completed,
        ResponseOutput::Embedding {
            values: vec![1.0],
            pooling: PoolingType::Mean,
            normalized: true,
        },
        "",
    );

    let error = generation_result_from_response(response).expect_err("embedding response");

    assert!(
        matches!(error, Error::RuntimeCommand(message) if message.contains("embedding output"))
    );
}

#[test]
fn embedding_response_maps_to_embedding_result() {
    let response = GenerateResponse::terminal(
        9,
        GenerateResponseStatus::Completed,
        ResponseOutput::Embedding {
            values: vec![0.6, 0.8],
            pooling: PoolingType::Mean,
            normalized: true,
        },
        "",
    );

    let result = embedding_result_from_response(response).expect("embedding result");

    assert_eq!(result.id, "9");
    assert_eq!(result.values, vec![0.6, 0.8]);
    assert_eq!(result.pooling, PoolingType::Mean);
    assert!(result.normalized);
}

#[test]
fn duration_millis_saturates_at_u64_max() {
    let oversized = Duration::from_millis(u64::MAX).saturating_add(Duration::from_millis(1));
    assert_eq!(duration_millis_u64(oversized), u64::MAX);
}
