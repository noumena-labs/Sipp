//! Tests the `config` module in `cogentlm-providers`.
//!
//! Covers stable provider labels, secret redaction/exposure, environment-backed
//! secret loading, and public config value construction with deterministic local
//! values and no provider calls.

use std::sync::Mutex;
use std::time::Duration;

use super::*;

static ENV_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn provider_kind_labels_are_stable() {
    assert_eq!(ProviderKind::Proxy.as_str(), "proxy");
    assert_eq!(ProviderKind::OpenAi.as_str(), "openai");
    assert_eq!(ProviderKind::Anthropic.as_str(), "anthropic");
}

#[test]
fn secret_string_exposes_value_without_debug_leakage() {
    let secret = SecretString::new("token-value");

    assert_eq!(secret.expose(), "token-value");
    assert!(!secret.is_empty());

    let debug = format!("{secret:?}");
    assert_eq!(debug, "SecretString([redacted])");
    assert!(!debug.contains("token-value"));

    let empty = SecretString::new("");
    assert!(empty.is_empty());
}

#[test]
fn secret_string_loads_from_environment() {
    let _guard = ENV_LOCK.lock().expect("env lock");
    let name = "COGENTLM_PROVIDER_TEST_SECRET_STRING_LOADS";
    std::env::set_var(name, "env-token");

    let secret = SecretString::from_env(name).expect("secret from env");

    assert_eq!(secret.expose(), "env-token");
    std::env::remove_var(name);
}

#[test]
fn public_config_values_are_constructible_and_comparable() {
    let openai = OpenAiConfig {
        api_key: SecretString::new("openai-token"),
        base_url: Some("http://localhost/openai".to_string()),
        timeout: Some(Duration::from_secs(3)),
    };
    assert_eq!(openai.clone(), openai);

    let anthropic = AnthropicConfig {
        api_key: SecretString::new("anthropic-token"),
        base_url: Some("http://localhost/anthropic".to_string()),
        version: Some("2023-06-01".to_string()),
        timeout: Some(Duration::from_secs(5)),
    };
    assert_eq!(anthropic.clone(), anthropic);

    let proxy = ProxyConfig {
        base_url: "http://localhost/proxy".to_string(),
        auth: ProviderAuth::Header {
            name: "x-api-key".to_string(),
            value: SecretString::new("proxy-token"),
        },
        protocol: ProxyProtocol::OpenAiCompatible,
        static_headers: vec![("x-static".to_string(), "yes".to_string())],
        timeout: Some(Duration::from_secs(7)),
    };
    assert_eq!(proxy.clone(), proxy);
}
