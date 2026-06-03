use super::BackendFileConfig;
use crate::{GatewayErrorKind, GatewayFileConfig};

#[tokio::test]
async fn gateway_config_builds_mock_alias_with_policy() {
    std::env::set_var("COGENTLM_TEST_GATEWAY_TOKEN", "test-token");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN"

[[auth.access.aliases]]
name = "mock"
operations = ["query"]

[limits]
max_request_bytes = 1024

[cors]
allowed_origins = ["https://app.example"]

[[aliases]]
name = "mock"
operations = ["query"]

[aliases.limits]
max_concurrent_requests = 2
max_requests_per_minute = 60
max_requests_total = 100

[aliases.backend]
kind = "mock"
text = "test: "
embedding_dimensions = 4
"#,
    )
    .expect("config");

    let server = config.build().await.expect("server config");
    assert_eq!(server.bind.to_string(), "127.0.0.1:8787");
    let _router = server.service.router().expect("router");
}

#[tokio::test]
async fn gateway_config_rejects_zero_limits() {
    std::env::set_var("COGENTLM_TEST_GATEWAY_TOKEN_ZERO", "test-token");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_ZERO"

[[aliases]]
name = "mock"
operations = ["query"]

[aliases.limits]
max_requests_total = 0

[aliases.backend]
kind = "mock"
"#,
    )
    .expect("config");

    let error = match config.build().await {
        Ok(_) => panic!("zero limit should fail"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        error.message,
        "max_requests_total must be greater than zero"
    );
}

#[test]
fn gateway_config_parses_local_cogent_engine_backend() {
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_LOCAL"

[[aliases]]
name = "local"
operations = ["query", "chat", "embed"]

[aliases.backend]
kind = "local_cogent_engine"
model_path = "models/local.gguf"

[aliases.backend.options]
context_key = "gateway-local"
grammar = "root ::= \"ok\""
json_schema = "{}"
embedding_context_key = "gateway-embed"
normalize_embeddings = false
"#,
    )
    .expect("config");

    let backend = &config.aliases[0].backend;
    match backend {
        BackendFileConfig::LocalCogentEngine {
            model_path,
            options,
            ..
        } => {
            assert_eq!(model_path, &std::path::PathBuf::from("models/local.gguf"));
            assert_eq!(options.context_key.as_deref(), Some("gateway-local"));
            assert_eq!(
                options.embedding_context_key.as_deref(),
                Some("gateway-embed")
            );
            assert_eq!(options.normalize_embeddings, Some(false));
        }
        _ => panic!("expected local CogentEngine backend"),
    }
}

#[test]
fn gateway_config_debug_redacts_static_provider_header_values() {
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_HEADER"

[[aliases]]
name = "compatible"
operations = ["chat"]

[aliases.backend]
kind = "open_ai_compatible"
model = "private-model"
base_url = "https://provider.example/v1"
timeout_ms = 1000

[aliases.backend.auth]
kind = "bearer"
token_env = "COGENTLM_TEST_PROVIDER_TOKEN"

[[aliases.backend.static_headers]]
name = "x-provider-secret"
value = "secret-header-value"
"#,
    )
    .expect("config");

    let debug = format!("{config:?}");

    assert!(debug.contains("x-provider-secret"));
    assert!(debug.contains("[redacted]"));
    assert!(!debug.contains("secret-header-value"));
}

#[test]
fn gateway_config_rejects_unknown_fields() {
    let error = GatewayFileConfig::from_toml_str(
        r#"
provider_api_key = "provider-secret"

[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_UNKNOWN"

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
    )
    .expect_err("unknown top-level field should fail");
    let message = error.message;
    assert!(message.contains("unknown field `provider_api_key`"));
    assert!(!message.contains("provider-secret"));

    let error = GatewayFileConfig::from_toml_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_UNKNOWN"

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
api_key_env = "OPENAI_API_KEY"
"#,
    )
    .expect_err("unknown backend field should fail");
    assert!(error.message.contains("unknown field `api_key_env`"));

    let error = GatewayFileConfig::from_toml_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_UNKNOWN"

[[aliases]]
name = "local"

[aliases.backend]
kind = "local_cogent_engine"
model_path = "models/local.gguf"

[aliases.backend.options]
contextkey = "misspelled"
"#,
    )
    .expect_err("unknown nested option should fail");
    assert!(error.message.contains("unknown field `contextkey`"));
}

#[test]
fn gateway_example_configs_parse() {
    for name in [
        "mock_gateway.toml",
        "local_cogent_engine_gateway.toml",
        "openai_compatible_gateway.toml",
        "openai_gateway.toml",
        "anthropic_gateway.toml",
    ] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("examples")
            .join(name);
        let contents = std::fs::read_to_string(&path).expect("example config");
        let _config = GatewayFileConfig::from_toml_str(&contents).expect("parse example config");
    }
}
