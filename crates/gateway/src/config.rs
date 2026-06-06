use std::{
    collections::BTreeSet,
    fmt,
    net::IpAddr,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use cogentlm_engine::engine::NativeRuntimeConfig;
use cogentlm_providers::{
    AnthropicAdapterConfig, OpenAiAdapterConfig, OpenAiCompatibleAdapterConfig,
    OpenAiCompatibleProtocol, ProviderAuth, ProviderTransport, SecretString,
};
use http::{HeaderName, HeaderValue, Uri};
use serde::Deserialize;

use crate::{
    server::validate_gateway_bearer_secret, GatewayAccess, GatewayAdapter, GatewayAlias,
    GatewayAliasLimits, GatewayError, GatewayErrorKind, GatewayRequestLimits, GatewayResult,
    LocalCogentEngineBackend, LocalCogentEngineOptions, MockBackend, Operation, OperationSet,
    ProviderGatewayBackend,
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

    /// Build the framework-agnostic gateway adapter.
    ///
    /// Server-specific settings such as bind address, bearer tokens, CORS, and
    /// body limits remain available on this parsed config for host servers, but
    /// they are not applied by the core adapter.
    pub async fn build(self) -> GatewayResult<GatewayAdapter> {
        self.build_adapter().await
    }

    /// Build the framework-agnostic gateway adapter.
    ///
    /// # Errors
    ///
    /// Returns an error when alias configuration or backend construction fails.
    pub async fn build_adapter(self) -> GatewayResult<GatewayAdapter> {
        let alias_names = validate_alias_configs(&self.aliases)?;
        self.auth.access.gateway_access(&alias_names)?;
        validate_alias_backend_configs(&self.aliases)?;
        let mut adapter = GatewayAdapter::new();
        for alias in self.aliases {
            adapter.add_alias(alias.build().await?)?;
        }
        Ok(adapter)
    }

    /// Return the access scope configured for the standalone server token.
    ///
    /// # Errors
    ///
    /// Returns an error when token access references unknown aliases.
    pub fn gateway_access(&self) -> GatewayResult<GatewayAccess> {
        let alias_names = validate_alias_configs(&self.aliases)?;
        self.auth.access.clone().gateway_access(&alias_names)
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
    /// Environment variable containing the standalone server admin token.
    pub admin_token_env: Option<String>,
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
    /// Convert this config into a gateway access scope.
    ///
    /// # Errors
    ///
    /// Returns an error when access entries are invalid or reference unknown
    /// aliases.
    pub fn gateway_access(
        self,
        configured_aliases: &BTreeSet<String>,
    ) -> GatewayResult<GatewayAccess> {
        match self.aliases {
            None => Ok(GatewayAccess::all()),
            Some(aliases) => {
                if aliases.is_empty() {
                    return Err(GatewayError::new(
                        GatewayErrorKind::InvalidRequest,
                        "token access aliases must not be empty",
                    ));
                }
                let mut names = BTreeSet::new();
                let mut access = Vec::with_capacity(aliases.len());
                for alias in aliases {
                    validate_config_name(&alias.name, "token access alias name")?;
                    if !names.insert(alias.name.clone()) {
                        return Err(GatewayError::new(
                            GatewayErrorKind::InvalidRequest,
                            "token access aliases must not contain duplicates",
                        ));
                    }
                    if !configured_aliases.contains(&alias.name) {
                        return Err(GatewayError::new(
                            GatewayErrorKind::InvalidRequest,
                            "token access alias is not configured",
                        ));
                    }
                    let operations = alias
                        .operations
                        .map(|operations| {
                            operation_set_from_config(operations, "token access operations")
                        })
                        .transpose()?
                        .unwrap_or_else(OperationSet::all);
                    access.push((alias.name, operations));
                }
                GatewayAccess::new(access)
            }
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
            .map(|operations| operation_set_from_config(operations, "alias operations"))
            .transpose()?
            .unwrap_or_else(OperationSet::all);
        let limits = self.limits.alias_limits()?;
        let backend = self.backend.backend().await?;
        GatewayAlias::new(self.name, operations, backend, limits)
    }
}

/// Gateway-wide service limit config.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServiceLimitsFileConfig {
    /// Maximum accepted request body bytes.
    pub max_request_bytes: Option<usize>,
    /// Maximum redacted request history entries retained by the server.
    pub history_capacity: Option<usize>,
}

impl ServiceLimitsFileConfig {
    /// Return the configured maximum request body size.
    ///
    /// # Errors
    ///
    /// Returns an error when the limit is zero.
    pub fn max_request_bytes(&self) -> GatewayResult<usize> {
        let max_request_bytes = self.max_request_bytes.unwrap_or(1 << 20);
        if max_request_bytes == 0 {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "max_request_bytes must be greater than zero",
            ));
        }
        Ok(max_request_bytes)
    }

    /// Return the configured request history capacity.
    ///
    /// # Errors
    ///
    /// Returns an error when the capacity is zero.
    pub fn history_capacity(&self) -> GatewayResult<usize> {
        let history_capacity = self.history_capacity.unwrap_or(200);
        if history_capacity == 0 {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "history_capacity must be greater than zero",
            ));
        }
        Ok(history_capacity)
    }
}

/// Alias-level limit config.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AliasLimitsFileConfig {
    /// Legacy maximum concurrent requests for this alias.
    pub max_concurrent_requests: Option<usize>,
    /// Legacy maximum requests per rolling minute for this alias.
    pub max_requests_per_minute: Option<u32>,
    /// Legacy maximum total requests allowed since gateway startup.
    pub max_requests_total: Option<u64>,
    /// Limits that apply across every caller.
    pub global: Option<RequestLimitsFileConfig>,
    /// Limits that apply independently to each caller ID.
    pub per_caller: Option<RequestLimitsFileConfig>,
    /// Maximum caller IDs tracked for per-caller limits.
    pub max_tracked_callers: Option<usize>,
}

/// Policy limits for one configured scope.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestLimitsFileConfig {
    /// Maximum concurrent requests for the scope.
    pub max_concurrent_requests: Option<usize>,
    /// Maximum requests per rolling minute for the scope.
    pub max_requests_per_minute: Option<u32>,
    /// Maximum total requests allowed since gateway startup.
    pub max_requests_total: Option<u64>,
}

impl AliasLimitsFileConfig {
    fn alias_limits(&self) -> GatewayResult<GatewayAliasLimits> {
        let legacy = RequestLimitsFileConfig {
            max_concurrent_requests: self.max_concurrent_requests,
            max_requests_per_minute: self.max_requests_per_minute,
            max_requests_total: self.max_requests_total,
        };
        let global = self.global.unwrap_or(legacy).request_limits();
        validate_request_limits(global)?;
        let per_caller = self.per_caller.map(RequestLimitsFileConfig::request_limits);
        if let Some(limits) = per_caller {
            validate_request_limits(limits)?;
        }
        let max_tracked_callers = self
            .max_tracked_callers
            .unwrap_or(GatewayAliasLimits::default().max_tracked_callers);
        if max_tracked_callers == 0 {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "max_tracked_callers must be greater than zero",
            ));
        }
        Ok(GatewayAliasLimits {
            global,
            per_caller,
            max_tracked_callers,
        })
    }
}

impl RequestLimitsFileConfig {
    const fn request_limits(self) -> GatewayRequestLimits {
        GatewayRequestLimits {
            max_concurrent_requests: self.max_concurrent_requests,
            max_requests_per_minute: self.max_requests_per_minute,
            max_requests_total: self.max_requests_total,
        }
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
    fn validate_before_secret_loading(&self) -> GatewayResult<()> {
        match self {
            Self::Mock { .. } => Ok(()),
            Self::LocalCogentEngine { options, .. } => options.validate_before_secret_loading(),
            Self::OpenAi {
                model, base_url, ..
            }
            | Self::Anthropic {
                model, base_url, ..
            } => {
                validate_config_name(model, "provider backend model")?;
                if let Some(base_url) = base_url.as_deref() {
                    validate_provider_base_url(base_url)?;
                }
                Ok(())
            }
            Self::OpenAiCompatible {
                model,
                base_url,
                auth,
                static_headers,
                ..
            } => {
                validate_config_name(model, "provider backend model")?;
                validate_provider_base_url(base_url)?;
                auth.validate_before_secret_loading()?;
                validate_provider_static_headers(static_headers)?;
                Ok(())
            }
        }
    }

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
            } => {
                validate_config_name(&model, "provider backend model")?;
                if let Some(base_url) = base_url.as_deref() {
                    validate_provider_base_url(base_url)?;
                }
                let transport = ProviderTransport::openai(OpenAiAdapterConfig {
                    api_key: env_secret(api_key_env)?,
                    base_url,
                    timeout: timeout_ms.map(Duration::from_millis),
                })
                .map_err(provider_config_error)?;
                Ok(std::sync::Arc::new(ProviderGatewayBackend::new(
                    model, transport,
                )?))
            }
            Self::Anthropic {
                model,
                api_key_env,
                base_url,
                version,
                timeout_ms,
            } => {
                validate_config_name(&model, "provider backend model")?;
                if let Some(base_url) = base_url.as_deref() {
                    validate_provider_base_url(base_url)?;
                }
                let transport = ProviderTransport::anthropic(AnthropicAdapterConfig {
                    api_key: env_secret(api_key_env)?,
                    base_url,
                    version,
                    timeout: timeout_ms.map(Duration::from_millis),
                })
                .map_err(provider_config_error)?;
                Ok(std::sync::Arc::new(ProviderGatewayBackend::new(
                    model, transport,
                )?))
            }
            Self::OpenAiCompatible {
                model,
                base_url,
                auth,
                static_headers,
                timeout_ms,
            } => {
                validate_config_name(&model, "provider backend model")?;
                validate_provider_base_url(&base_url)?;
                let static_headers = provider_static_headers(static_headers)?;
                let transport =
                    ProviderTransport::openai_compatible(OpenAiCompatibleAdapterConfig {
                        base_url,
                        auth: auth.provider_auth()?,
                        protocol: OpenAiCompatibleProtocol::OpenAiCompatible,
                        static_headers,
                        timeout: timeout_ms.map(Duration::from_millis),
                    })
                    .map_err(provider_config_error)?;
                Ok(std::sync::Arc::new(ProviderGatewayBackend::new(
                    model, transport,
                )?))
            }
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
    fn validate_before_secret_loading(&self) -> GatewayResult<()> {
        if self
            .context_key
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "local CogentEngine context_key must not be empty",
            ));
        }
        if self
            .embedding_context_key
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "local CogentEngine embedding_context_key must not be empty",
            ));
        }
        Ok(())
    }

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
    fn validate_before_secret_loading(&self) -> GatewayResult<()> {
        match self {
            Self::Bearer { .. } => Ok(()),
            Self::Header { name, .. } => {
                validate_provider_header_name(name, "provider auth header name")
            }
        }
    }

    fn provider_auth(self) -> GatewayResult<ProviderAuth> {
        match self {
            Self::Bearer { token_env } => Ok(ProviderAuth::Bearer(env_secret(token_env)?)),
            Self::Header { name, value_env } => {
                validate_provider_header_name(&name, "provider auth header name")?;
                Ok(ProviderAuth::Header {
                    name,
                    value: env_secret(value_env)?,
                })
            }
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

fn env_secret(name: impl AsRef<str>) -> GatewayResult<SecretString> {
    let name = name.as_ref();
    let value = std::env::var(name).map_err(|error| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("failed to read secret env var {name}: {error}"),
        )
    })?;
    if value.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("secret env var {name} must not be empty"),
        ));
    }
    validate_gateway_bearer_secret(&value, &format!("secret env var {name}"))?;
    Ok(SecretString::new(value))
}

fn provider_config_error(error: cogentlm_providers::ProviderError) -> GatewayError {
    GatewayError::new(
        GatewayErrorKind::InvalidRequest,
        format!("invalid provider config: {}", error.message),
    )
}

fn validate_provider_base_url(base_url: &str) -> GatewayResult<()> {
    let trimmed = base_url.trim();
    if trimmed.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url must not be empty",
        ));
    }
    if trimmed != base_url {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url must not contain surrounding whitespace",
        ));
    }
    if base_url.contains('#') {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url must not include query or fragment",
        ));
    }

    let uri = base_url.parse::<Uri>().map_err(|_| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url is invalid",
        )
    })?;
    let Some(scheme) = uri.scheme_str() else {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url must be an absolute http(s) URL",
        ));
    };
    let Some(authority) = uri.authority() else {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url must be an absolute http(s) URL",
        ));
    };
    if !matches!(scheme, "http" | "https") {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url must be an absolute http(s) URL",
        ));
    }
    if authority.as_str().contains('@') {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url must not include userinfo",
        ));
    }
    if uri
        .path_and_query()
        .and_then(http::uri::PathAndQuery::query)
        .is_some()
    {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url must not include query or fragment",
        ));
    }
    if scheme == "http" && !uri.host().is_some_and(is_loopback_host) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider base_url must use HTTPS unless it targets loopback",
        ));
    }
    Ok(())
}

fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    match host
        .trim_matches(|character| character == '[' || character == ']')
        .parse::<IpAddr>()
    {
        Ok(address) => address.is_loopback(),
        Err(_) => false,
    }
}

fn provider_static_headers(headers: Vec<HeaderFileConfig>) -> GatewayResult<Vec<(String, String)>> {
    let mut output = Vec::with_capacity(headers.len());
    for header in headers {
        validate_provider_static_header(&header)?;
        output.push((header.name, header.value));
    }
    Ok(output)
}

fn validate_provider_static_headers(headers: &[HeaderFileConfig]) -> GatewayResult<()> {
    for header in headers {
        validate_provider_static_header(header)?;
    }
    Ok(())
}

fn validate_provider_static_header(header: &HeaderFileConfig) -> GatewayResult<()> {
    validate_provider_header_name(&header.name, "provider static header name")?;
    validate_provider_header_value(&header.value, "provider static header value")
}

fn validate_provider_header_name(name: &str, field: &'static str) -> GatewayResult<()> {
    validate_config_name(name, field)?;
    HeaderName::from_bytes(name.as_bytes()).map_err(|_| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} is not a valid HTTP header name"),
        )
    })?;
    Ok(())
}

fn validate_provider_header_value(value: &str, field: &'static str) -> GatewayResult<()> {
    HeaderValue::from_str(value).map_err(|_| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} is not a valid HTTP header value"),
        )
    })?;
    Ok(())
}

fn validate_alias_configs(aliases: &[AliasFileConfig]) -> GatewayResult<BTreeSet<String>> {
    if aliases.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "gateway config requires at least one alias",
        ));
    }
    let mut names = BTreeSet::new();
    for alias in aliases {
        validate_config_name(&alias.name, "alias name")?;
        if !names.insert(alias.name.clone()) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway aliases must not contain duplicates",
            ));
        }
        if let Some(operations) = &alias.operations {
            operation_set_from_config(operations.iter().copied(), "alias operations")?;
        }
        alias.limits.alias_limits()?;
    }
    Ok(names)
}

fn validate_alias_backend_configs(aliases: &[AliasFileConfig]) -> GatewayResult<()> {
    for alias in aliases {
        alias.backend.validate_before_secret_loading()?;
    }
    Ok(())
}

fn validate_config_name(value: &str, field: &'static str) -> GatewayResult<()> {
    if value.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not be empty"),
        ));
    }
    if value.trim() != value {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not contain surrounding whitespace"),
        ));
    }
    Ok(())
}

fn operation_set_from_config(
    operations: impl IntoIterator<Item = OperationFileConfig>,
    field: &'static str,
) -> GatewayResult<OperationSet> {
    let mut seen = BTreeSet::new();
    let mut values = Vec::new();
    for operation in operations {
        let operation = operation.operation();
        if !seen.insert(operation) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!("{field} must not contain duplicates"),
            ));
        }
        values.push(operation);
    }
    if values.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not be empty"),
        ));
    }
    Ok(OperationSet::new(values))
}

fn validate_request_limits(limits: GatewayRequestLimits) -> GatewayResult<()> {
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
    Ok(())
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
