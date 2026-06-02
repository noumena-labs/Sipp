//! Integration tests for the `cogentlm-client` crate-level public_api surface.
//!
//! Covers endpoint resolution, remote configuration, facade validation, and run wrappers with deterministic fakes rather than a live local engine.

use cogentlm_client::{
    CogentClient, CogentError, CogentQueryRequest, CogentTextOptions, EndpointRef, LocalTextOptions,
};

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
