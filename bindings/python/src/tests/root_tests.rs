//! Unit tests for the parent module.

use super::*;
use cogentlm_core::{TokenBatch, TokenEmissionStats, TokenUsage};
use cogentlm_engine::engine::RequestStats;
use pyo3::types::PyDict;

#[test]
fn py_i64_to_u32_rejects_out_of_range_seed_values() {
    assert_eq!(py_i64_to_u32(0).expect("zero"), 0);
    assert_eq!(py_i64_to_u32(i64::from(u32::MAX)).expect("max"), u32::MAX);
    assert!(py_i64_to_u32(-1).is_err());
    assert!(py_i64_to_u32(i64::from(u32::MAX) + 1).is_err());
}

#[test]
fn py_finite_f32_rejects_non_finite_values() {
    assert_eq!(py_finite_f32(0.5, "temperature").expect("finite"), 0.5);
    assert!(py_finite_f32(f32::NAN, "temperature").is_err());
    assert!(py_finite_f32(f32::INFINITY, "temperature").is_err());
    assert!(py_optional_finite_f32(Some(f32::NEG_INFINITY), "temperature").is_err());
}

#[test]
fn py_sampling_config_rejects_non_finite_float_inputs() {
    let config = PySamplingRuntimeConfig::new(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Some(f32::NAN),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        false,
        false,
        None,
        true,
    );
    assert!(config.is_err());
}

#[test]
fn py_placement_config_rejects_non_finite_tensor_split() {
    let config = PyModelPlacementConfig::new(
        None,
        None,
        None,
        None,
        Some(vec![1.0, f32::INFINITY]),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    );
    assert!(config.is_err());
}

#[test]
fn py_chat_message_requires_valid_role() {
    assert!(PyChatMessage::new("tool".to_string(), "bad".to_string()).is_err());

    let message = PyChatMessage::new("assistant".to_string(), "ok".to_string())
        .expect("chat message")
        .to_core()
        .expect("core message");

    assert_eq!(message.role, ChatRole::Assistant);
    assert_eq!(message.content, "ok");
}

#[test]
fn py_text_options_validate_finite_float_fields() {
    let options = PyCogentTextOptions::new(
        Some(8),
        Some(0.25),
        Some(0.9),
        Some(vec!["stop".to_string()]),
    )
    .expect("text options")
    .to_core();

    assert_eq!(options.max_tokens, Some(8));
    assert_eq!(options.temperature, Some(0.25));
    assert_eq!(options.top_p, Some(0.9));
    assert_eq!(options.stop, vec!["stop"]);

    assert!(PyCogentTextOptions::new(None, Some(f32::NAN), None, None).is_err());
}

#[test]
fn py_to_json_converts_nested_values_and_rejects_non_finite_float() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let dict = PyDict::new_bound(py);
        dict.set_item("flag", true).expect("flag");
        dict.set_item("items", vec![1, 2, 3]).expect("items");

        let json = py_to_json(dict.as_any()).expect("json");

        assert_eq!(json["flag"], serde_json::json!(true));
        assert_eq!(json["items"], serde_json::json!([1, 2, 3]));
        assert!(py_to_json(f64::NAN.into_py(py).bind(py)).is_err());
    });
}

#[test]
fn py_response_helpers_map_endpoint_usage_and_stats() {
    pyo3::prepare_freethreaded_python();
    Python::with_gil(|py| {
        let endpoint = endpoint_ref_to_dict(
            py,
            CoreEndpointRef::Remote {
                id: "remote".to_string(),
            },
        )
        .expect("endpoint")
        .bind(py)
        .downcast::<PyDict>()
        .expect("endpoint dict");
        assert_eq!(
            endpoint
                .get_item("kind")
                .expect("kind")
                .expect("kind value")
                .extract::<String>()
                .expect("kind string"),
            "remote"
        );

        let usage = token_usage_to_dict(
            py,
            TokenUsage {
                input_tokens: Some(1),
                output_tokens: Some(2),
                total_tokens: Some(3),
            },
        )
        .expect("usage")
        .bind(py)
        .downcast::<PyDict>()
        .expect("usage dict");
        assert_eq!(
            usage
                .get_item("total_tokens")
                .expect("total")
                .expect("total value")
                .extract::<u32>()
                .expect("total int"),
            3
        );

        let batch = token_batch_to_dict(
            py,
            TokenBatch {
                request_id: "req".to_string(),
                stream_id: 1,
                sequence_start: 2,
                text: "hello".to_string(),
                frame_count: 1,
                byte_count: 5,
                stats: TokenEmissionStats {
                    frames_sent: 1,
                    bytes_sent: 5,
                    batches_sent: 1,
                },
            },
        )
        .expect("batch")
        .bind(py)
        .downcast::<PyDict>()
        .expect("batch dict");
        assert_eq!(
            batch
                .get_item("text")
                .expect("text")
                .expect("text value")
                .extract::<String>()
                .expect("text string"),
            "hello"
        );

        let stats = request_stats_to_dict(
            py,
            RequestStats {
                input_tokens: 1,
                output_tokens: 2,
                ..RequestStats::default()
            },
        )
        .expect("stats")
        .bind(py)
        .downcast::<PyDict>()
        .expect("stats dict");
        assert_eq!(
            stats
                .get_item("cache_source")
                .expect("cache source")
                .expect("cache source value")
                .extract::<String>()
                .expect("cache source string"),
            "none"
        );
    });
}
