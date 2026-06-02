//! Tests the `remote` module in `cogentlm-client`.
//!
//! Covers endpoint resolution, remote configuration, facade validation, and run wrappers with deterministic fakes rather than a live local engine.

use std::time::Duration;

use cogentlm_providers::{ProviderAuth, ProviderKind};

use super::*;

#[test]
fn remote_secret_debug_redacts_raw_value() {
    let secret = RemoteSecret::new("sk-test-secret");
    let rendered = format!("{secret:?}");

    assert!(rendered.contains("[redacted]"));
    assert!(!rendered.contains("sk-test-secret"));
    assert_eq!(secret.expose(), "sk-test-secret");
}

#[test]
fn remote_auth_maps_to_provider_auth_without_losing_fields() {
    let bearer = RemoteAuth::Bearer(RemoteSecret::new("bearer-token")).to_provider();
    assert!(matches!(
        bearer,
        ProviderAuth::Bearer(secret) if secret.expose() == "bearer-token"
    ));

    let header = RemoteAuth::Header {
        name: "X-Api-Key".to_string(),
        value: RemoteSecret::new("header-token"),
    }
    .to_provider();
    assert!(matches!(
        header,
        ProviderAuth::Header { name, value }
            if name == "X-Api-Key" && value.expose() == "header-token"
    ));
}

#[test]
fn remote_constructors_preserve_defaults_and_overrides() {
    let openai = RemoteConfig::openai("gpt-test", "openai-key");
    let RemoteConfig::OpenAi(openai) = openai else {
        panic!("openai constructor should create OpenAi config");
    };
    assert_eq!(openai.model, "gpt-test");
    assert_eq!(openai.api_key.expose(), "openai-key");
    assert!(openai.base_url.is_none());
    assert!(openai.timeout.is_none());

    let mut anthropic = match RemoteConfig::anthropic("claude-test", "anthropic-key") {
        RemoteConfig::Anthropic(config) => config,
        _ => panic!("anthropic constructor should create Anthropic config"),
    };
    anthropic.version = Some("2023-06-01".to_string());
    anthropic.timeout = Some(Duration::from_secs(10));
    assert_eq!(anthropic.model, "claude-test");
    assert_eq!(anthropic.api_key.expose(), "anthropic-key");
    assert_eq!(anthropic.version.as_deref(), Some("2023-06-01"));
    assert_eq!(anthropic.timeout, Some(Duration::from_secs(10)));
}

#[test]
fn proxy_config_builds_proxy_transport() {
    let (model, transport) = RemoteConfig::proxy(
        "proxy-model",
        "http://localhost:11434",
        RemoteAuth::Bearer(RemoteSecret::new("proxy-token")),
    )
    .build()
    .expect("proxy transport");

    assert_eq!(model, "proxy-model");
    assert_eq!(transport.kind(), ProviderKind::Proxy);
}
