use std::{
    fmt,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use cogentlm_engine::engine::NativeRuntimeConfig;
use cogentlm_gateway_providers::{
    AnthropicAdapterConfig, GatewayAdapterTransport, OpenAiAdapterConfig,
    OpenAiCompatibleAdapterConfig, OpenAiCompatibleProtocol, ProviderAuth, SecretString,
};
use serde::Deserialize;

use crate::{
    GatewayAccess, GatewayAlias, GatewayAliasLimits, GatewayError, GatewayErrorKind, GatewayResult,
    GatewayService, GatewayServiceLimits, GatewayState, GatewayToken, LocalCogentEngineBackend,
    LocalCogentEngineOptions, MockBackend, Operation, OperationSet, ProviderGatewayBackend,
};

/// Runtime server configuration loaded from `gateway.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayFileConfig {
    /// Server binding configuration.
    pub server: ServerFileConfig,
    /// Gateway authentication configuration.
    pub auth: AuthFileConfig,
    /// Gateway-wide limits.
    #[serde(default)]
    pub limits: ServiceLimitsFileConfig,
    /// Browser CORS configuration.
    #[serde(default)]
    pub cors: CorsFileConfig,
    /// Alias definitions.
    pub aliases: Vec<AliasFileConfig>,
}

impl GatewayFileConfig {
    /// Parse a gateway configuration from TOML.
    pub fn from_toml_str(contents: &str) -> GatewayResult<Self> {
        toml::from_str(contents).map_err(|error| {
            GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!(
                    "failed to parse gateway config: {}",
                    toml_error_message(error)
                ),
            )
        })
    }

    /// Load a gateway configuration from a TOML file.
    pub fn from_path(path: impl AsRef<Path>) -> GatewayResult<Self> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path).map_err(|error| {
            GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!("failed to read gateway config {}: {error}", path.display()),
            )
        })?;
        toml::from_str(&contents).map_err(|error| {
            GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!(
                    "failed to parse gateway config {}: {}",
                    path.display(),
                    toml_error_message(error)
                ),
            )
        })
    }

    /// Build the runnable gateway server configuration.
    pub async fn build(self) -> GatewayResult<GatewayServerConfig> {
        let token = env_secret(&self.auth.token_env)?;
        let mut state = GatewayState::with_tokens([GatewayToken::new(
            token.expose().to_string(),
            self.auth.access.gateway_access(),
        )]);
        for alias in self.aliases {
            state.add_alias(alias.build().await?)?;
        }
        Ok(GatewayServerConfig {
            bind: self.server.bind,
            service: GatewayService::new(
                state,
                self.cors.allowed_origins,
                self.limits.service_limits()?,
            ),
        })
    }
}

/// Server listener config.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerFileConfig {
    /// Address and port to bind, for example `127.0.0.1:8787`.
    pub bind: SocketAddr,
}

/// Gateway auth config.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AuthFileConfig {
    /// Environment variable containing the gateway bearer token.
    pub token_env: String,
    /// Access scope for the configured gateway token.
    #[serde(default)]
    pub access: TokenAccessFileConfig,
}

/// Gateway token access scope from config.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenAccessFileConfig {
    /// Allowed aliases for this token. Omitted means every alias.
    pub aliases: Option<Vec<TokenAliasAccessFileConfig>>,
}

impl TokenAccessFileConfig {
    fn gateway_access(self) -> GatewayAccess {
        match self.aliases {
            None => GatewayAccess::all(),
            Some(aliases) => GatewayAccess::new(aliases.into_iter().map(|alias| {
                let operations = alias
                    .operations
                    .map(|operations| {
                        OperationSet::new(
                            operations.into_iter().map(OperationFileConfig::operation),
                        )
                    })
                    .unwrap_or_else(OperationSet::all);
                (alias.name, operations)
            })),
        }
    }
}

/// Token access for a single alias.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenAliasAccessFileConfig {
    /// Public alias name.
    pub name: String,
    /// Allowed operations for this token on the alias. Omitted means every operation.
    pub operations: Option<Vec<OperationFileConfig>>,
}

/// Gateway CORS config.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorsFileConfig {
    /// Exact browser origins allowed to call the gateway.
    #[serde(default)]
    pub allowed_origins: Vec<String>,
}

/// One public gateway alias.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasFileConfig {
    /// Public alias exposed to clients as `model`.
    pub name: String,
    /// Enabled operations. Omitted means every operation.
    pub operations: Option<Vec<OperationFileConfig>>,
    /// Backend for this alias.
    pub backend: BackendFileConfig,
    /// Alias-level policy limits.
    #[serde(default)]
    pub limits: AliasLimitsFileConfig,
}

impl AliasFileConfig {
    async fn build(self) -> GatewayResult<GatewayAlias> {
        let operations = self
            .operations
            .map(|operations| {
                OperationSet::new(operations.into_iter().map(OperationFileConfig::operation))
            })
            .unwrap_or_else(OperationSet::all);
        Ok(GatewayAlias::new(
            self.name,
            operations,
            self.backend.backend().await?,
            self.limits.alias_limits()?,
        ))
    }
}

/// Gateway-wide service limit config.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceLimitsFileConfig {
    /// Maximum accepted request body bytes.
    pub max_request_bytes: Option<usize>,
}

impl ServiceLimitsFileConfig {
    fn service_limits(self) -> GatewayResult<GatewayServiceLimits> {
        let max_request_bytes = self
            .max_request_bytes
            .unwrap_or(GatewayServiceLimits::default().max_request_bytes);
        if max_request_bytes == 0 {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "max_request_bytes must be greater than zero",
            ));
        }
        Ok(GatewayServiceLimits { max_request_bytes })
    }
}

/// Alias-level limit config.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasLimitsFileConfig {
    /// Maximum concurrent requests for this alias.
    pub max_concurrent_requests: Option<usize>,
    /// Maximum requests per rolling minute for this alias.
    pub max_requests_per_minute: Option<u32>,
    /// Maximum total requests allowed since gateway startup.
    pub max_requests_total: Option<u64>,
}

impl AliasLimitsFileConfig {
    fn alias_limits(self) -> GatewayResult<GatewayAliasLimits> {
        let limits = GatewayAliasLimits {
            max_concurrent_requests: self.max_concurrent_requests,
            max_requests_per_minute: self.max_requests_per_minute,
            max_requests_total: self.max_requests_total,
        };
        if matches!(limits.max_concurrent_requests, Some(0)) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "max_concurrent_requests must be greater than zero",
            ));
        }
        if matches!(limits.max_requests_per_minute, Some(0)) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "max_requests_per_minute must be greater than zero",
            ));
        }
        if matches!(limits.max_requests_total, Some(0)) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "max_requests_total must be greater than zero",
            ));
        }
        Ok(limits)
    }
}

/// Operation name in gateway config.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationFileConfig {
    /// Raw prompt operation.
    Query,
    /// Chat operation.
    Chat,
    /// Embedding operation.
    Embed,
}

impl OperationFileConfig {
    const fn operation(self) -> Operation {
        match self {
            Self::Query => Operation::Query,
            Self::Chat => Operation::Chat,
            Self::Embed => Operation::Embed,
        }
    }
}

/// Backend config for a gateway alias.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum BackendFileConfig {
    /// Deterministic mock backend.
    Mock {
        /// Prefix returned before request text.
        #[serde(default = "default_mock_text")]
        text: String,
        /// Number of embedding dimensions returned by mock embed.
        #[serde(default = "default_mock_embedding_dimensions")]
        embedding_dimensions: usize,
    },
    /// Hosted-local CogentEngine backend.
    LocalCogentEngine {
        /// Local GGUF model path loaded by the gateway process.
        model_path: PathBuf,
        /// Native runtime configuration.
        #[serde(default)]
        runtime: Box<NativeRuntimeConfig>,
        /// Request defaults owned by the gateway alias.
        #[serde(default)]
        options: LocalCogentEngineFileOptions,
    },
    /// OpenAI backend.
    OpenAi {
        /// Private upstream model name.
        model: String,
        /// Environment variable containing the OpenAI API key.
        api_key_env: String,
        /// Optional provider base URL.
        base_url: Option<String>,
        /// Optional request timeout in milliseconds.
        timeout_ms: Option<u64>,
    },
    /// Anthropic backend.
    Anthropic {
        /// Private upstream model name.
        model: String,
        /// Environment variable containing the Anthropic API key.
        api_key_env: String,
        /// Optional provider base URL.
        base_url: Option<String>,
        /// Optional Anthropic API version.
        version: Option<String>,
        /// Optional request timeout in milliseconds.
        timeout_ms: Option<u64>,
    },
    /// OpenAI-compatible backend.
    OpenAiCompatible {
        /// Private upstream model name.
        model: String,
        /// Provider-compatible base URL.
        base_url: String,
        /// Auth config.
        auth: OpenAiCompatibleAuthFileConfig,
        /// Static provider headers owned by gateway config.
        #[serde(default)]
        static_headers: Vec<HeaderFileConfig>,
        /// Optional request timeout in milliseconds.
        timeout_ms: Option<u64>,
    },
}

impl BackendFileConfig {
    async fn backend(self) -> GatewayResult<std::sync::Arc<dyn crate::GatewayBackend>> {
        match self {
            Self::Mock {
                text,
                embedding_dimensions,
            } => Ok(std::sync::Arc::new(MockBackend::new(
                text,
                embedding_dimensions,
            ))),
            Self::LocalCogentEngine {
                model_path,
                runtime,
                options,
            } => Ok(std::sync::Arc::new(
                LocalCogentEngineBackend::load(model_path, *runtime, options.local_options())
                    .await?,
            )),
            Self::OpenAi {
                model,
                api_key_env,
                base_url,
                timeout_ms,
            } => Ok(std::sync::Arc::new(ProviderGatewayBackend::new(
                model,
                GatewayAdapterTransport::openai(OpenAiAdapterConfig {
                    api_key: env_secret(api_key_env)?,
                    base_url,
                    timeout: timeout_ms.map(Duration::from_millis),
                })
                .map_err(provider_config_error)?,
            ))),
            Self::Anthropic {
                model,
                api_key_env,
                base_url,
                version,
                timeout_ms,
            } => Ok(std::sync::Arc::new(ProviderGatewayBackend::new(
                model,
                GatewayAdapterTransport::anthropic(AnthropicAdapterConfig {
                    api_key: env_secret(api_key_env)?,
                    base_url,
                    version,
                    timeout: timeout_ms.map(Duration::from_millis),
                })
                .map_err(provider_config_error)?,
            ))),
            Self::OpenAiCompatible {
                model,
                base_url,
                auth,
                static_headers,
                timeout_ms,
            } => Ok(std::sync::Arc::new(ProviderGatewayBackend::new(
                model,
                GatewayAdapterTransport::openai_compatible(OpenAiCompatibleAdapterConfig {
                    base_url,
                    auth: auth.provider_auth()?,
                    protocol: OpenAiCompatibleProtocol::OpenAiCompatible,
                    static_headers: static_headers
                        .into_iter()
                        .map(|header| (header.name, header.value))
                        .collect(),
                    timeout: timeout_ms.map(Duration::from_millis),
                })
                .map_err(provider_config_error)?,
            ))),
        }
    }
}

/// Request defaults for a hosted-local CogentEngine backend.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalCogentEngineFileOptions {
    /// Context key used for query and chat requests.
    pub context_key: Option<String>,
    /// Grammar applied to query and chat requests.
    pub grammar: Option<String>,
    /// JSON schema applied to query and chat requests.
    pub json_schema: Option<String>,
    /// Context key used for embedding requests.
    pub embedding_context_key: Option<String>,
    /// Whether embedding vectors should be L2-normalized.
    pub normalize_embeddings: Option<bool>,
}

impl LocalCogentEngineFileOptions {
    fn local_options(self) -> LocalCogentEngineOptions {
        let defaults = LocalCogentEngineOptions::default();
        LocalCogentEngineOptions {
            context_key: self.context_key.unwrap_or(defaults.context_key),
            grammar: self.grammar.unwrap_or_default(),
            json_schema: self.json_schema.unwrap_or_default(),
            embedding_context_key: self.embedding_context_key,
            normalize_embeddings: self
                .normalize_embeddings
                .unwrap_or(defaults.normalize_embeddings),
        }
    }
}

/// Auth config for an OpenAI-compatible upstream.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
#[serde(deny_unknown_fields)]
pub enum OpenAiCompatibleAuthFileConfig {
    /// Bearer token loaded from an environment variable.
    Bearer {
        /// Environment variable name.
        token_env: String,
    },
    /// Custom header value loaded from an environment variable.
    Header {
        /// Header name.
        name: String,
        /// Environment variable containing the header value.
        value_env: String,
    },
}

impl OpenAiCompatibleAuthFileConfig {
    fn provider_auth(self) -> GatewayResult<ProviderAuth> {
        match self {
            Self::Bearer { token_env } => Ok(ProviderAuth::Bearer(env_secret(token_env)?)),
            Self::Header { name, value_env } => Ok(ProviderAuth::Header {
                name,
                value: env_secret(value_env)?,
            }),
        }
    }
}

/// Static header entry for an OpenAI-compatible upstream.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeaderFileConfig {
    /// Header name.
    pub name: String,
    /// Header value.
    pub value: String,
}

impl fmt::Debug for HeaderFileConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeaderFileConfig")
            .field("name", &self.name)
            .field("value", &"[redacted]")
            .finish()
    }
}

/// Runnable gateway service configuration.
pub struct GatewayServerConfig {
    /// Socket address the gateway binds to.
    pub bind: SocketAddr,
    /// Gateway service.
    pub service: GatewayService,
}

fn env_secret(name: impl AsRef<str>) -> GatewayResult<SecretString> {
    let name = name.as_ref();
    let value = std::env::var(name).map_err(|error| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("failed to read secret env var {name}: {error}"),
        )
    })?;
    if value.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("secret env var {name} must not be empty"),
        ));
    }
    Ok(SecretString::new(value))
}

fn provider_config_error(error: cogentlm_gateway_providers::ProviderError) -> GatewayError {
    GatewayError::new(
        GatewayErrorKind::InvalidRequest,
        format!("invalid provider config: {}", error.message),
    )
}

fn toml_error_message(error: toml::de::Error) -> String {
    error.message().to_string()
}

fn default_mock_text() -> String {
    "mock: ".to_string()
}

const fn default_mock_embedding_dimensions() -> usize {
    8
}

#[cfg(test)]
#[path = "tests/config_tests.rs"]
mod config_tests;
