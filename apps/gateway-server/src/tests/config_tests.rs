//! Tests server configuration validation, production defaults, and the shipped
//! side-effect-free configuration fixture.

use std::{path::Path, time::Duration};

use crate::config::{GatewayServerConfig, TokenConfig};

#[test]
fn check_rejects_missing_aliases_without_reading_environment() {
    let config = GatewayServerConfig {
        tokens: vec![TokenConfig {
            env: "MISSING_TEST_TOKEN".to_string(),
            caller: "test".to_string(),
            access: Vec::new(),
        }],
        ..GatewayServerConfig::default()
    };

    assert!(config.validate().is_err());
}

#[test]
fn production_defaults_use_long_stream_drain_windows() {
    let config = GatewayServerConfig::default();

    assert_eq!(config.drain_timeout(), Duration::from_secs(120));
    assert_eq!(config.force_close_timeout(), Duration::from_secs(5));
}

#[test]
fn shipped_production_config_parses_without_environment_access() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("config/production.toml");

    GatewayServerConfig::from_path(&path).expect("production config should be valid");
}

#[test]
fn alias_endpoint_schema_rejects_unknown_fields() {
    let source = r#"
        [[tokens]]
        env = "COGENTLM_GATEWAY_TOKEN"
        caller = "test"

        [[aliases]]
        name = "local"
        type = "local"
        model = "model.gguf"
        unexpected = true
    "#;

    assert!(toml::from_str::<GatewayServerConfig>(source).is_err());
}
