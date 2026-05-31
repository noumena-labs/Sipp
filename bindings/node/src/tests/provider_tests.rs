//! Provider binding tests for the parent module.

use super::*;
use serde_json::json;

#[test]
fn provider_proxy_config_maps_static_headers() {
    let config = ProviderProxyConfig {
        base_url: "http://localhost".to_string(),
        auth: ProviderAuthConfig {
            bearer: Some("token".to_string()),
            header: None,
        },
        protocol: None,
        static_headers: Some(vec![ProviderStaticHeaderConfig {
            name: "x-cogent-test".to_string(),
            value: "yes".to_string(),
        }]),
        timeout_ms: Some(1500),
    };

    let core = config.to_core().expect("provider proxy config");

    assert_eq!(
        core.static_headers,
        vec![("x-cogent-test".to_string(), "yes".to_string())]
    );
    assert_eq!(core.protocol, CoreProxyProtocol::OpenAiCompatible);
    assert_eq!(core.timeout, Some(Duration::from_millis(1500)));
    match core.auth {
        CoreProviderAuth::Bearer(secret) => assert_eq!(secret.expose(), "token"),
        CoreProviderAuth::Header { .. } => panic!("expected bearer auth"),
    }
}

#[test]
fn provider_anthropic_config_maps_core_fields() {
    let config = ProviderAnthropicConfig {
        api_key: "token".to_string(),
        base_url: Some("http://localhost".to_string()),
        version: Some("2023-06-01".to_string()),
        timeout_ms: Some(1500),
    };

    let core = config.to_core();

    assert_eq!(core.api_key.expose(), "token");
    assert_eq!(core.base_url.as_deref(), Some("http://localhost"));
    assert_eq!(core.version.as_deref(), Some("2023-06-01"));
    assert_eq!(core.timeout, Some(Duration::from_millis(1500)));
}

#[test]
fn provider_options_must_be_json_object() {
    assert!(provider_options_or_empty(Some(json!("not-object"))).is_err());

    let options = provider_options_or_empty(Some(json!({ "seed": 7 }))).expect("options");

    assert_eq!(options.get("seed"), Some(&json!(7)));
}

#[test]
fn provider_generation_options_reject_invalid_numbers() {
    let options = ProviderGenerationOptions {
        max_tokens: Some(0),
        temperature: None,
        top_p: None,
        stop: None,
    };
    assert!(options.to_core().is_err());

    let options = ProviderGenerationOptions {
        max_tokens: None,
        temperature: Some(f64::NAN),
        top_p: None,
        stop: None,
    };
    assert!(options.to_core().is_err());
}

#[test]
fn provider_chat_response_maps_usage_and_metadata() {
    let response = CoreProviderChatResponse {
        result: CoreProviderTextOutput {
            text: "hi".to_string(),
            finish_reason: cogentlm_engine::engine::FinishReason::Stop,
        },
        usage: Some(CoreTokenUsage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            total_tokens: Some(3),
        }),
        metadata: CoreProviderResponseMetadata {
            provider: cogentlm_providers::ProviderKind::Proxy,
            model: "proxy-model".to_string(),
            request_id: Some("req-1".to_string()),
            response_id: Some("resp-1".to_string()),
            finish_reason_raw: Some("stop".to_string()),
            raw: json!({ "id": "resp-1" }),
        },
    };

    let mapped = provider_chat_response_to_node(response);

    assert_eq!(mapped.result.text, "hi");
    assert_eq!(mapped.result.finish_reason, "stop");
    assert_eq!(mapped.usage.expect("usage").total_tokens, Some(3));
    assert_eq!(mapped.metadata.provider, "proxy");
    assert_eq!(mapped.metadata.request_id.as_deref(), Some("req-1"));
    assert_eq!(mapped.metadata.finish_reason_raw.as_deref(), Some("stop"));
    assert_eq!(mapped.metadata.raw["id"], "resp-1");
}

#[test]
fn provider_embedding_response_maps_usage_and_metadata() {
    let response = CoreProviderEmbeddingResponse {
        result: CoreProviderEmbeddingOutput {
            values: vec![0.25, -0.5],
        },
        usage: Some(CoreTokenUsage {
            input_tokens: Some(2),
            output_tokens: None,
            total_tokens: Some(2),
        }),
        metadata: CoreProviderResponseMetadata {
            provider: cogentlm_providers::ProviderKind::Proxy,
            model: "proxy-model".to_string(),
            request_id: Some("req-embed".to_string()),
            response_id: None,
            finish_reason_raw: None,
            raw: json!({ "object": "list" }),
        },
    };

    let mapped = provider_embedding_response_to_node(response);

    assert_eq!(mapped.result.values, vec![0.25, -0.5]);
    assert_eq!(mapped.usage.expect("usage").total_tokens, Some(2));
    assert_eq!(mapped.metadata.model, "proxy-model");
    assert_eq!(mapped.metadata.request_id.as_deref(), Some("req-embed"));
    assert_eq!(mapped.metadata.raw["object"], "list");
}

#[test]
fn provider_error_maps_to_node_status_and_message() {
    let error = CoreProviderError::new(
        CoreProviderErrorKind::InvalidRequest,
        cogentlm_providers::ProviderKind::Proxy,
        "bad request",
    );

    assert_eq!(
        provider_error_message(&error),
        "proxy provider error (invalid_request): bad request"
    );
    assert_eq!(
        provider_error_status(CoreProviderErrorKind::InvalidRequest),
        Status::InvalidArg
    );
    assert_eq!(
        provider_error_status(CoreProviderErrorKind::RateLimited),
        Status::GenericFailure
    );
}
