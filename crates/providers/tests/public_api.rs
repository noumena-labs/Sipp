//! Integration tests for the `cogentlm-providers` crate-level public_api surface.
//!
//! Covers stable exported provider labels, secret redaction, and construction of
//! public config/request/response/model value types with deterministic values
//! and no provider calls.

use std::time::Duration;

use cogentlm_core::{ChatMessage, ChatRole, FinishReason};
use cogentlm_providers::{
    AnthropicConfig, CapabilitySupport, OpenAiConfig, ProviderAuth, ProviderCapabilities,
    ProviderChatRequest, ProviderEmbedRequest, ProviderEmbeddingOutput, ProviderEmbeddingResponse,
    ProviderError, ProviderErrorKind, ProviderGenerateRequest, ProviderGenerationOptions,
    ProviderKind, ProviderModel, ProviderOptions, ProviderResponse, ProviderResponseMetadata,
    ProviderTextOutput, ProxyConfig, ProxyProtocol, SecretString, TokenUsage,
};

#[test]
fn provider_kind_has_stable_wire_labels() {
    assert_eq!(ProviderKind::Proxy.as_str(), "proxy");
    assert_eq!(ProviderKind::OpenAi.as_str(), "openai");
    assert_eq!(ProviderKind::Anthropic.as_str(), "anthropic");
}

#[test]
fn secret_debug_output_is_redacted() {
    let secret = SecretString::new("real-token");
    let debug = format!("{secret:?}");

    assert!(debug.contains("redacted"));
    assert!(!debug.contains("real-token"));
    assert_eq!(secret.expose(), "real-token");
}

#[test]
fn public_config_request_response_and_model_types_are_constructible() {
    let options = ProviderGenerationOptions {
        max_tokens: Some(16),
        temperature: Some(0.25),
        top_p: Some(0.9),
        stop: vec!["END".to_string()],
    };
    let provider_options: ProviderOptions = [("seed".to_string(), serde_json::json!(7))]
        .into_iter()
        .collect();

    let chat = ProviderChatRequest {
        model: "model-a".to_string(),
        messages: vec![ChatMessage::new(ChatRole::User, "hello")],
        options: options.clone(),
        provider_options: provider_options.clone(),
    };
    assert_eq!(chat.messages[0].content, "hello");

    let generate = ProviderGenerateRequest {
        model: "model-a".to_string(),
        prompt: "tell me".to_string(),
        options,
        provider_options: provider_options.clone(),
    };
    assert_eq!(generate.prompt, "tell me");

    let embed = ProviderEmbedRequest {
        model: "model-a".to_string(),
        input: "embed me".to_string(),
        provider_options,
    };
    assert_eq!(embed.input, "embed me");

    let text_response: ProviderResponse<ProviderTextOutput> = ProviderResponse {
        result: ProviderTextOutput {
            text: "done".to_string(),
            finish_reason: FinishReason::Stop,
        },
        usage: Some(TokenUsage {
            input_tokens: Some(1),
            output_tokens: Some(2),
            total_tokens: Some(3),
        }),
        metadata: ProviderResponseMetadata {
            provider: ProviderKind::Proxy,
            model: "model-a".to_string(),
            request_id: Some("req-1".to_string()),
            response_id: Some("resp-1".to_string()),
            finish_reason_raw: Some("stop".to_string()),
            raw: serde_json::json!({ "id": "resp-1" }),
        },
    };
    assert_eq!(text_response.result.text, "done");

    let embedding_response: ProviderEmbeddingResponse = ProviderResponse {
        result: ProviderEmbeddingOutput {
            values: vec![0.25, -0.5],
        },
        usage: None,
        metadata: ProviderResponseMetadata {
            provider: ProviderKind::OpenAi,
            model: "embed-model".to_string(),
            request_id: None,
            response_id: None,
            finish_reason_raw: None,
            raw: serde_json::Value::Null,
        },
    };
    assert_eq!(embedding_response.result.values.len(), 2);

    let model = ProviderModel {
        id: "model-a".to_string(),
        provider: ProviderKind::Proxy,
        display_name: Some("Model A".to_string()),
        capabilities: ProviderCapabilities {
            chat: CapabilitySupport::Supported,
            generate: CapabilitySupport::Unknown,
            embeddings: CapabilitySupport::Unsupported,
            token_emission: CapabilitySupport::Supported,
        },
        context_window: Some(8192),
        max_output_tokens: Some(1024),
        raw: serde_json::json!({ "id": "model-a" }),
    };
    assert_eq!(model.capabilities.chat, CapabilitySupport::Supported);
}

#[test]
fn public_config_and_error_types_are_constructible() {
    let openai = OpenAiConfig {
        api_key: SecretString::new("openai-token"),
        base_url: Some("http://localhost/openai".to_string()),
        timeout: Some(Duration::from_secs(1)),
    };
    assert_eq!(openai.base_url.as_deref(), Some("http://localhost/openai"));

    let anthropic = AnthropicConfig {
        api_key: SecretString::new("anthropic-token"),
        base_url: Some("http://localhost/anthropic".to_string()),
        version: Some("2023-06-01".to_string()),
        timeout: None,
    };
    assert_eq!(anthropic.version.as_deref(), Some("2023-06-01"));

    let proxy = ProxyConfig {
        base_url: "http://localhost/proxy".to_string(),
        auth: ProviderAuth::Bearer(SecretString::new("proxy-token")),
        protocol: ProxyProtocol::OpenAiCompatible,
        static_headers: vec![("x-test".to_string(), "yes".to_string())],
        timeout: None,
    };
    assert_eq!(proxy.static_headers.len(), 1);

    let err = ProviderError::new(
        ProviderErrorKind::UnsupportedFeature,
        ProviderKind::Anthropic,
        "embeddings are unsupported",
    );
    assert_eq!(err.kind.as_str(), "unsupported_feature");
    assert_eq!(err.provider, ProviderKind::Anthropic);
}
