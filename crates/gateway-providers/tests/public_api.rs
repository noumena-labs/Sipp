use cogentlm_gateway_providers::{
    OpenAiCompatibleAdapterConfig, OpenAiCompatibleProtocol, ProviderAuth, ProviderKind,
    SecretString,
};

#[test]
fn provider_kind_has_stable_wire_labels() {
    assert_eq!(ProviderKind::OpenAiCompatible.as_str(), "openai_compatible");
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
fn openai_compatible_config_debug_redacts_static_header_values() {
    let config = OpenAiCompatibleAdapterConfig {
        base_url: "https://provider.example".to_string(),
        auth: ProviderAuth::Bearer(SecretString::new("gateway-token")),
        protocol: OpenAiCompatibleProtocol::OpenAiCompatible,
        static_headers: vec![("x-provider-secret".to_string(), "secret-value".to_string())],
        timeout: None,
    };
    let debug = format!("{config:?}");

    assert!(debug.contains("x-provider-secret"));
    assert!(debug.contains("[redacted]"));
    assert!(!debug.contains("gateway-token"));
    assert!(!debug.contains("secret-value"));
}
