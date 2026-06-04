//! Tests the `remote` module in `cogentlm-client`.
//!
//! Covers remote gateway configuration, alias validation, secret redaction, and
//! transport construction without sending network requests.

use std::time::Duration;

use super::*;
use crate::{CogentError, RemoteErrorKind};

fn build_error(config: RemoteGatewayConfig) -> CogentError {
    match config.build() {
        Ok(_) => panic!("remote gateway config should fail"),
        Err(error) => error,
    }
}

#[test]
fn remote_secret_debug_redacts_raw_value() {
    let secret = RemoteSecret::new("sk-test-secret");
    let rendered = format!("{secret:?}");

    assert!(rendered.contains("[redacted]"));
    assert!(!rendered.contains("sk-test-secret"));
    assert_eq!(secret.expose(), "sk-test-secret");
}

#[test]
fn remote_gateway_config_builds_gateway_transport() {
    let (alias, _transport) = RemoteGatewayConfig {
        alias: "chat-pro".to_string(),
        base_url: "http://localhost:11434".to_string(),
        token: RemoteSecret::new("gateway-token"),
        timeout: Some(Duration::from_secs(5)),
    }
    .build()
    .expect("gateway transport");

    assert_eq!(alias, "chat-pro");
}

#[test]
fn remote_gateway_config_rejects_invalid_aliases_before_transport_config() {
    let blank = build_error(RemoteGatewayConfig {
        alias: "   ".to_string(),
        base_url: "https://user:secret@gateway.example".to_string(),
        token: RemoteSecret::new("gateway-token"),
        timeout: None,
    });
    assert!(matches!(
        blank,
        CogentError::InvalidRequest(message) if message == "remote alias must not be empty"
    ));

    let surrounded = build_error(RemoteGatewayConfig {
        alias: " chat-pro ".to_string(),
        base_url: "https://user:secret@gateway.example".to_string(),
        token: RemoteSecret::new("gateway-token"),
        timeout: None,
    });
    assert!(matches!(
        surrounded,
        CogentError::InvalidRequest(message)
            if message == "remote alias must not contain surrounding whitespace"
    ));
}

#[test]
fn remote_gateway_config_maps_transport_validation_errors() {
    let empty_token = build_error(RemoteGatewayConfig {
        alias: "chat-pro".to_string(),
        base_url: "https://gateway.example".to_string(),
        token: RemoteSecret::new(""),
        timeout: None,
    });
    assert!(matches!(
        empty_token,
        CogentError::Remote(remote) if remote.kind == RemoteErrorKind::Authentication
    ));

    let zero_timeout = build_error(RemoteGatewayConfig {
        alias: "chat-pro".to_string(),
        base_url: "https://gateway.example".to_string(),
        token: RemoteSecret::new("gateway-token"),
        timeout: Some(Duration::ZERO),
    });
    assert!(matches!(
        zero_timeout,
        CogentError::Remote(remote) if remote.kind == RemoteErrorKind::InvalidRequest
    ));
}
