//! Integration tests for the `cogentlm-client` crate-level public_api surface.
//!
//! Covers public request envelopes, facade error behavior, and provider-gated
//! remote configuration types without loading local models or calling remotes.

use cogentlm_client::{
    CogentClient, CogentError, CogentQueryRequest, CogentTextOptions, EndpointRef, LocalTextOptions,
};
#[cfg(feature = "remote")]
use cogentlm_client::{RemoteError, RemoteErrorKind, RemoteGatewayConfig, RemoteSecret};

#[test]
fn request_envelopes_are_publicly_constructible() {
    let request = CogentQueryRequest {
        endpoint: Some(EndpointRef::Local {
            id: "local".to_string(),
        }),
        prompt: "hello".to_string(),
        options: CogentTextOptions {
            max_tokens: Some(8),
            temperature: Some(0.0),
            top_p: Some(1.0),
            stop: vec!["</s>".to_string()],
        },
        local: LocalTextOptions {
            context_key: Some("ctx".to_string()),
            ..LocalTextOptions::default()
        },
        ..CogentQueryRequest::default()
    };

    assert_eq!(request.prompt, "hello");
    assert_eq!(request.options.max_tokens, Some(8));
    assert_eq!(request.local.context_key.as_deref(), Some("ctx"));
}

#[test]
fn empty_client_reports_no_supported_endpoint() {
    let client = CogentClient::new();
    let error = futures::executor::block_on(client.query(CogentQueryRequest {
        prompt: "hello".to_string(),
        ..CogentQueryRequest::default()
    }))
    .expect_err("empty client should fail");

    assert!(matches!(
        error,
        CogentError::NoSupportedEndpoint { operation: "query" }
    ));
}

#[cfg(feature = "remote")]
#[test]
fn remote_configuration_types_are_publicly_constructible() {
    let config = RemoteGatewayConfig {
        alias: "model".to_string(),
        base_url: "http://localhost:11434".to_string(),
        token: RemoteSecret::new("secret"),
        timeout: None,
    };

    assert_eq!(config.alias, "model");
    assert_eq!(config.base_url, "http://localhost:11434");
    assert!(format!("{:?}", config.token).contains("[redacted]"));

    let error = RemoteError::new(RemoteErrorKind::Timeout, "slow");
    assert_eq!(error.kind, RemoteErrorKind::Timeout);
    assert_eq!(error.message, "slow");
}
