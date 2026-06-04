//! Unit tests for the parent module.

use super::*;
use serde_json::json;

#[test]
fn i64_to_u32_rejects_out_of_range_seed_values() {
    assert_eq!(i64_to_u32(0, "seed").expect("zero"), 0);
    assert_eq!(
        i64_to_u32(i64::from(u32::MAX), "seed").expect("max"),
        u32::MAX
    );
    assert!(i64_to_u32(-1, "seed").is_err());
    assert!(i64_to_u32(i64::from(u32::MAX) + 1, "seed").is_err());
}

#[test]
fn f64_to_f32_rejects_non_finite_and_out_of_range_values() {
    assert_eq!(f64_to_f32(0.5, "temperature").expect("finite"), 0.5_f32);
    assert!(f64_to_f32(f64::NAN, "temperature").is_err());
    assert!(f64_to_f32(f64::INFINITY, "temperature").is_err());
    assert!(f64_to_f32(f64::from(f32::MAX) * 2.0, "temperature").is_err());
    assert!(f64_to_f32(f64::from(f32::MIN) * 2.0, "temperature").is_err());
}

#[test]
fn sampling_config_rejects_non_finite_float_inputs() {
    let config = SamplingRuntimeConfig {
        temperature: Some(f64::NAN),
        ..Default::default()
    };
    assert!(config.to_core().is_err());
}

#[test]
fn placement_config_rejects_non_finite_tensor_split() {
    let config = ModelPlacementConfig {
        tensor_split: Some(vec![1.0, f64::INFINITY]),
        ..Default::default()
    };
    assert!(config.to_core().is_err());
}

#[test]
fn u64_to_js_safe_number_clamps_at_number_safe_integer() {
    assert_eq!(u64_to_js_safe_number(42), 42.0);
    assert_eq!(
        u64_to_js_safe_number(JS_MAX_SAFE_INTEGER_U64),
        JS_MAX_SAFE_INTEGER_F64
    );
    assert_eq!(u64_to_js_safe_number(u64::MAX), JS_MAX_SAFE_INTEGER_F64);
}

#[test]
fn i64_to_js_safe_number_clamps_negative_and_large_values() {
    assert_eq!(i64_to_js_safe_number(-1), 0.0);
    assert_eq!(i64_to_js_safe_number(42), 42.0);
    assert_eq!(i64_to_js_safe_number(i64::MAX), JS_MAX_SAFE_INTEGER_F64);
}

#[test]
fn finite_nonnegative_f64_to_u64_requires_safe_integer() {
    assert_eq!(
        finite_nonnegative_f64_to_u64(JS_MAX_SAFE_INTEGER_F64, "bytes").expect("max safe"),
        JS_MAX_SAFE_INTEGER_U64
    );
    assert!(finite_nonnegative_f64_to_u64(1.5, "bytes").is_err());
    assert!(finite_nonnegative_f64_to_u64(-1.0, "bytes").is_err());
    assert!(finite_nonnegative_f64_to_u64(f64::NAN, "bytes").is_err());
    assert!(finite_nonnegative_f64_to_u64(JS_MAX_SAFE_INTEGER_F64 + 2.0, "bytes").is_err());
}

#[test]
fn finite_nonnegative_f64_to_u64_parses_exact_integer_boundaries() {
    assert_eq!(finite_nonnegative_f64_to_u64(0.0, "bytes").unwrap(), 0);
    assert_eq!(finite_nonnegative_f64_to_u64(42.0, "bytes").unwrap(), 42);
    assert_eq!(
        finite_nonnegative_f64_to_u64(JS_MAX_SAFE_INTEGER_F64, "bytes").unwrap(),
        JS_MAX_SAFE_INTEGER_U64
    );
}

#[test]
fn endpoint_ref_maps_valid_kinds_and_rejects_unknown_kind() {
    let local = EndpointRef {
        kind: "local".to_string(),
        id: "local-a".to_string(),
    }
    .to_core()
    .expect("local endpoint");
    assert!(matches!(
        local,
        CoreEndpointRef::Local { id } if id == "local-a"
    ));

    let remote = EndpointRef {
        kind: "remote".to_string(),
        id: "remote-a".to_string(),
    }
    .to_core()
    .expect("remote endpoint");
    assert!(matches!(
        remote,
        CoreEndpointRef::Remote { id } if id == "remote-a"
    ));

    let invalid = EndpointRef {
        kind: "browser".to_string(),
        id: "bad".to_string(),
    };
    assert!(invalid.to_core().is_err());
}

#[test]
fn text_options_validate_finite_float_fields() {
    let options = CogentTextOptions {
        max_tokens: Some(4),
        temperature: Some(0.25),
        top_p: Some(0.9),
        stop: Some(vec!["stop".to_string()]),
    }
    .to_core()
    .expect("text options");

    assert_eq!(options.max_tokens, Some(4));
    assert_eq!(options.temperature, Some(0.25));
    assert_eq!(options.top_p, Some(0.9));
    assert_eq!(options.stop, vec!["stop"]);

    let bad = CogentTextOptions {
        temperature: Some(f64::NAN),
        ..Default::default()
    };
    assert!(bad.to_core().is_err());
}

#[test]
fn chat_messages_require_non_empty_valid_roles() {
    assert!(chat_messages_to_core(Vec::new()).is_err());
    assert!(chat_messages_to_core(vec![ChatMessage {
        role: "tool".to_string(),
        content: "bad".to_string(),
    }])
    .is_err());

    let messages = chat_messages_to_core(vec![ChatMessage {
        role: "user".to_string(),
        content: "hello".to_string(),
    }])
    .expect("messages");

    assert_eq!(messages[0].role, CoreChatRole::User);
    assert_eq!(messages[0].content, "hello");
}

#[test]
fn query_request_maps_endpoint_options_and_remote_options() {
    let request = CogentQueryRequest {
        endpoint: Some(EndpointRef {
            kind: "remote".to_string(),
            id: "openai".to_string(),
        }),
        prompt: "hello".to_string(),
        options: Some(CogentTextOptions {
            max_tokens: Some(8),
            temperature: Some(0.0),
            top_p: None,
            stop: None,
        }),
        local: Some(LocalTextOptions {
            context_key: Some("ctx".to_string()),
            grammar: Some("root ::= \"ok\"".to_string()),
            json_schema: None,
            sampling: None,
            media: None,
        }),
        remote_options: Some(json!({ "seed": 7 })),
        emit_tokens: Some(true),
    }
    .to_core()
    .expect("query request");

    assert!(matches!(
        request.endpoint,
        Some(CoreEndpointRef::Remote { id }) if id == "openai"
    ));
    assert_eq!(request.prompt, "hello");
    assert_eq!(request.options.max_tokens, Some(8));
    assert_eq!(request.local.context_key.as_deref(), Some("ctx"));
    assert_eq!(request.remote_options.get("seed"), Some(&json!(7)));
    assert!(request.emit_tokens);
}

#[test]
fn request_remote_options_must_be_json_objects() {
    let request = CogentEmbedRequest {
        endpoint: None,
        input: "hello".to_string(),
        local: None,
        remote_options: Some(json!(["bad"])),
    };

    assert!(request.to_core().is_err());
}
