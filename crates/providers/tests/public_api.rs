use cogentlm_providers::{ProviderKind, SecretString};

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
