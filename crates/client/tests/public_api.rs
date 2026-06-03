//! Integration tests for the `cogentlm-client` crate-level public_api surface.
//!
//! Covers public request envelopes, facade error behavior, and provider-gated
//! remote configuration types without loading local models or calling remotes.

use cogentlm_client::{
    CogentClient, CogentError, CogentQueryRequest, CogentTextOptions, EndpointRef, LocalTextOptions,
};
#[cfg(feature = "providers")]
use cogentlm_client::{RemoteAuth, RemoteConfig, RemoteError, RemoteErrorKind, RemoteKind};

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

#[cfg(feature = "providers")]
#[test]
fn remote_configuration_types_are_publicly_constructible() {
    let config = RemoteConfig::proxy(
        "model",
        "http://localhost:11434",
        RemoteAuth::Bearer(cogentlm_client::RemoteSecret::new("secret")),
    );

    let RemoteConfig::Proxy(proxy) = config else {
        panic!("proxy constructor should create proxy config");
    };
    assert_eq!(proxy.model, "model");
    assert_eq!(proxy.base_url, "http://localhost:11434");

    let error = RemoteError::new(RemoteErrorKind::Timeout, RemoteKind::Proxy, "slow");
    assert_eq!(error.kind, RemoteErrorKind::Timeout);
    assert_eq!(error.remote_kind, RemoteKind::Proxy);
    assert_eq!(error.message, "slow");
}
