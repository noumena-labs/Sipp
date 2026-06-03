use super::{env_secret, BackendFileConfig};
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
    let _router = server.service.router();
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

#[tokio::test]
async fn gateway_config_validates_service_limit_before_loading_secret_env() {
    std::env::remove_var("COGENTLM_TEST_MISSING_GATEWAY_TOKEN_ZERO_SERVICE_LIMIT");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_MISSING_GATEWAY_TOKEN_ZERO_SERVICE_LIMIT"

[limits]
max_request_bytes = 0

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
    )
    .expect("config");

    let error = match config.build().await {
        Ok(_) => panic!("service limit should fail before loading gateway token"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(error.message, "max_request_bytes must be greater than zero");
}

#[tokio::test]
async fn gateway_config_validates_alias_limit_before_loading_secrets() {
    std::env::remove_var("COGENTLM_TEST_MISSING_GATEWAY_TOKEN_ZERO_ALIAS_LIMIT");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_MISSING_GATEWAY_TOKEN_ZERO_ALIAS_LIMIT"

[[aliases]]
name = "mock"

[aliases.limits]
max_concurrent_requests = 0

[aliases.backend]
kind = "mock"
"#,
    )
    .expect("config");
    let error = match config.build().await {
        Ok(_) => panic!("alias limit should fail before loading gateway token"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        error.message,
        "max_concurrent_requests must be greater than zero"
    );

    std::env::set_var(
        "COGENTLM_TEST_GATEWAY_TOKEN_ZERO_ALIAS_LIMIT_PROVIDER",
        "test-token",
    );
    std::env::remove_var("COGENTLM_TEST_MISSING_PROVIDER_TOKEN_ZERO_ALIAS_LIMIT");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_ZERO_ALIAS_LIMIT_PROVIDER"

[[aliases]]
name = "provider"

[aliases.limits]
max_requests_per_minute = 0

[aliases.backend]
kind = "open_ai"
model = "private-model"
api_key_env = "COGENTLM_TEST_MISSING_PROVIDER_TOKEN_ZERO_ALIAS_LIMIT"
"#,
    )
    .expect("config");
    let error = match config.build().await {
        Ok(_) => panic!("alias limit should fail before loading provider token"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        error.message,
        "max_requests_per_minute must be greater than zero"
    );
}

#[tokio::test]
async fn gateway_config_rejects_empty_alias_list() {
    std::env::set_var("COGENTLM_TEST_GATEWAY_TOKEN_NO_ALIAS", "test-token");
    let config: GatewayFileConfig = toml::from_str(
        r#"
aliases = []

[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_NO_ALIAS"
"#,
    )
    .expect("config");

    let error = match config.build().await {
        Ok(_) => panic!("empty alias list should fail"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(error.message, "gateway config requires at least one alias");
}

#[tokio::test]
async fn gateway_config_rejects_invalid_alias_names_and_operation_sets() {
    std::env::set_var("COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ALIAS", "test-token");

    for (config, message) in [
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ALIAS"

[[aliases]]
name = " mock "

[aliases.backend]
kind = "mock"
"#,
            "alias name must not contain surrounding whitespace",
        ),
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ALIAS"

[[aliases]]
name = "mock"
operations = []

[aliases.backend]
kind = "mock"
"#,
            "alias operations must not be empty",
        ),
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ALIAS"

[[aliases]]
name = "mock"
operations = ["query", "query"]

[aliases.backend]
kind = "mock"
"#,
            "alias operations must not contain duplicates",
        ),
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ALIAS"

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
            "gateway aliases must not contain duplicates",
        ),
    ] {
        let config: GatewayFileConfig = toml::from_str(config).expect("config");
        let error = match config.build().await {
            Ok(_) => panic!("{message} should fail"),
            Err(error) => error,
        };
        assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
        assert_eq!(error.message, message);
    }
}

#[tokio::test]
async fn gateway_config_rejects_invalid_token_access_entries() {
    std::env::set_var("COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ACCESS", "test-token");

    for (config, message) in [
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ACCESS"

[[auth.access.aliases]]
name = " mock "

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
            "token access alias name must not contain surrounding whitespace",
        ),
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ACCESS"

[[auth.access.aliases]]
name = "mock"

[[auth.access.aliases]]
name = "mock"

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
            "token access aliases must not contain duplicates",
        ),
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ACCESS"

[[auth.access.aliases]]
name = "mock"
operations = []

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
            "token access operations must not be empty",
        ),
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ACCESS"

[[auth.access.aliases]]
name = "mock"
operations = ["query", "query"]

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
            "token access operations must not contain duplicates",
        ),
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_ACCESS"

[[auth.access.aliases]]
name = "unknown"

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
            "token access alias is not configured",
        ),
    ] {
        let config: GatewayFileConfig = toml::from_str(config).expect("config");
        let error = match config.build().await {
            Ok(_) => panic!("{message} should fail"),
            Err(error) => error,
        };
        assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
        assert_eq!(error.message, message);
    }
}

#[tokio::test]
async fn gateway_config_validates_token_access_before_loading_secret_env() {
    std::env::remove_var("COGENTLM_TEST_MISSING_GATEWAY_TOKEN_INVALID_ACCESS");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_MISSING_GATEWAY_TOKEN_INVALID_ACCESS"

[[auth.access.aliases]]
name = "unknown"

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
    )
    .expect("config");

    let error = match config.build().await {
        Ok(_) => panic!("invalid token access should fail before loading secrets"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(error.message, "token access alias is not configured");
}

#[tokio::test]
async fn gateway_config_validates_provider_model_before_loading_secret_env() {
    std::env::remove_var("COGENTLM_TEST_MISSING_PROVIDER_TOKEN_INVALID_MODEL");
    std::env::set_var("COGENTLM_TEST_GATEWAY_TOKEN_INVALID_MODEL", "test-token");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_MODEL"

[[aliases]]
name = "provider"

[aliases.backend]
kind = "open_ai"
model = " provider-model "
api_key_env = "COGENTLM_TEST_MISSING_PROVIDER_TOKEN_INVALID_MODEL"
"#,
    )
    .expect("config");

    let error = match config.build().await {
        Ok(_) => panic!("invalid provider model should fail before loading provider secrets"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        error.message,
        "provider backend model must not contain surrounding whitespace"
    );
}

#[tokio::test]
async fn gateway_config_validates_provider_model_before_loading_gateway_token_env() {
    std::env::remove_var("COGENTLM_TEST_MISSING_GATEWAY_TOKEN_INVALID_PROVIDER_MODEL");
    std::env::remove_var("COGENTLM_TEST_MISSING_PROVIDER_TOKEN_INVALID_PROVIDER_MODEL");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_MISSING_GATEWAY_TOKEN_INVALID_PROVIDER_MODEL"

[[aliases]]
name = "provider"

[aliases.backend]
kind = "open_ai"
model = " provider-model "
api_key_env = "COGENTLM_TEST_MISSING_PROVIDER_TOKEN_INVALID_PROVIDER_MODEL"
"#,
    )
    .expect("config");

    let error = match config.build().await {
        Ok(_) => panic!("invalid provider model should fail before loading gateway token"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        error.message,
        "provider backend model must not contain surrounding whitespace"
    );
}

#[tokio::test]
async fn gateway_config_validates_provider_base_url_before_loading_secret_env() {
    std::env::remove_var("COGENTLM_TEST_MISSING_PROVIDER_TOKEN_INVALID_BASE_URL");
    std::env::set_var("COGENTLM_TEST_GATEWAY_TOKEN_INVALID_BASE_URL", "test-token");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_BASE_URL"

[[aliases]]
name = "provider"

[aliases.backend]
kind = "open_ai_compatible"
model = "private-model"
base_url = "https://user:provider-secret@provider.example/v1"

[aliases.backend.auth]
kind = "bearer"
token_env = "COGENTLM_TEST_MISSING_PROVIDER_TOKEN_INVALID_BASE_URL"
"#,
    )
    .expect("config");

    let error = match config.build().await {
        Ok(_) => panic!("invalid provider base URL should fail before loading provider secrets"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(error.message, "provider base_url must not include userinfo");
    assert!(!error.to_string().contains("provider-secret"));
}

#[tokio::test]
async fn gateway_config_validates_cors_before_loading_gateway_token_env() {
    std::env::remove_var("COGENTLM_TEST_MISSING_GATEWAY_TOKEN_INVALID_CORS");
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_MISSING_GATEWAY_TOKEN_INVALID_CORS"

[cors]
allowed_origins = [" https://app.example"]

[[aliases]]
name = "mock"

[aliases.backend]
kind = "mock"
"#,
    )
    .expect("config");

    let error = match config.build().await {
        Ok(_) => panic!("invalid CORS origin should fail before loading gateway token"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        error.message,
        "invalid CORS origin  https://app.example: surrounding whitespace is not allowed"
    );
}

#[tokio::test]
async fn gateway_config_validates_local_backend_options_before_loading_gateway_token_env() {
    std::env::remove_var("COGENTLM_TEST_MISSING_GATEWAY_TOKEN_INVALID_LOCAL_OPTIONS");

    for (config, message) in [
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_MISSING_GATEWAY_TOKEN_INVALID_LOCAL_OPTIONS"

[[aliases]]
name = "local"

[aliases.backend]
kind = "local_cogent_engine"
model_path = "models/local.gguf"

[aliases.backend.options]
context_key = " "
"#,
            "local CogentEngine context_key must not be empty",
        ),
        (
            r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_MISSING_GATEWAY_TOKEN_INVALID_LOCAL_OPTIONS"

[[aliases]]
name = "local"

[aliases.backend]
kind = "local_cogent_engine"
model_path = "models/local.gguf"

[aliases.backend.options]
embedding_context_key = " "
"#,
            "local CogentEngine embedding_context_key must not be empty",
        ),
    ] {
        let config: GatewayFileConfig = toml::from_str(config).expect("config");
        let error = match config.build().await {
            Ok(_) => {
                panic!("invalid local backend options should fail before loading gateway token")
            }
            Err(error) => error,
        };
        assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
        assert_eq!(error.message, message);
    }
}

#[tokio::test]
async fn gateway_config_validates_static_provider_headers_before_loading_secret_env() {
    std::env::remove_var("COGENTLM_TEST_MISSING_PROVIDER_TOKEN_INVALID_STATIC_HEADER");
    std::env::set_var(
        "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_STATIC_HEADER",
        "test-token",
    );
    let config: GatewayFileConfig = toml::from_str(
        r#"
[server]
bind = "127.0.0.1:8787"

[auth]
token_env = "COGENTLM_TEST_GATEWAY_TOKEN_INVALID_STATIC_HEADER"

[[aliases]]
name = "provider"

[aliases.backend]
kind = "open_ai_compatible"
model = "private-model"
base_url = "https://provider.example/v1"

[aliases.backend.auth]
kind = "bearer"
token_env = "COGENTLM_TEST_MISSING_PROVIDER_TOKEN_INVALID_STATIC_HEADER"

[[aliases.backend.static_headers]]
name = "bad header"
value = "provider-secret-header"
"#,
    )
    .expect("config");

    let error = match config.build().await {
        Ok(_) => panic!("invalid static header should fail before loading provider secrets"),
        Err(error) => error,
    };
    assert_eq!(error.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        error.message,
        "provider static header name is not a valid HTTP header name"
    );
    assert!(!error.to_string().contains("provider-secret-header"));
}

#[test]
fn gateway_secret_env_rejects_blank_or_whitespace_values() {
    std::env::set_var("COGENTLM_TEST_BLANK_SECRET", " \t ");
    let blank = env_secret("COGENTLM_TEST_BLANK_SECRET").expect_err("blank secret");
    assert_eq!(blank.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        blank.message,
        "secret env var COGENTLM_TEST_BLANK_SECRET must not be empty"
    );

    std::env::set_var("COGENTLM_TEST_WHITESPACE_SECRET", "secret token");
    let whitespace = env_secret("COGENTLM_TEST_WHITESPACE_SECRET").expect_err("whitespace secret");
    assert_eq!(whitespace.kind, GatewayErrorKind::InvalidRequest);
    assert_eq!(
        whitespace.message,
        "secret env var COGENTLM_TEST_WHITESPACE_SECRET must not contain whitespace"
    );
    assert!(!format!("{whitespace:?}").contains("secret token"));
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
