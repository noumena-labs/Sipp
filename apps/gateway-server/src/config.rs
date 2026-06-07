use std::{
    collections::{BTreeSet, HashMap},
    net::SocketAddr,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{bail, Context};
use cogentlm_client::{
    AnthropicProviderConfig, CogentClient, EndpointDescriptor, OpenAiCompatibleProviderConfig,
    OpenAiProviderConfig, ProviderAuthConfig, ProviderEndpointConfig, ProviderSecret,
};
use cogentlm_engine::engine::NativeRuntimeConfig;
use cogentlm_gateway::{
    CogentClientExecutor, GatewayAccess, GatewayAdapter, GatewayAlias, GatewayAliasLimits,
    GatewayCaller, GatewayRequestLimits, Operation, OperationSet,
};
use serde::Deserialize;

const DEFAULT_MAX_REQUEST_BYTES: usize = 1 << 20;
const DEFAULT_DRAIN_TIMEOUT_SECONDS: u64 = 120;
const DEFAULT_FORCE_CLOSE_TIMEOUT_SECONDS: u64 = 5;

/// Standalone gateway configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GatewayServerConfig {
    /// Public inference listener.
    pub public_bind: SocketAddr,
    /// Management listener for probes and metrics.
    pub management_bind: SocketAddr,
    /// Maximum JSON request body size.
    pub max_request_bytes: usize,
    /// Graceful drain deadline.
    pub drain_timeout_seconds: u64,
    /// Final window for terminal stream errors before force-close.
    pub force_close_timeout_seconds: u64,
    /// Optional browser origins accepted by the public listener.
    pub allowed_origins: Vec<String>,
    /// Scoped bearer tokens loaded from environment variables.
    pub tokens: Vec<TokenConfig>,
    /// Required endpoint aliases.
    pub aliases: Vec<AliasConfig>,
}

impl Default for GatewayServerConfig {
    fn default() -> Self {
        Self {
            public_bind: SocketAddr::from(([0, 0, 0, 0], 8080)),
            management_bind: SocketAddr::from(([0, 0, 0, 0], 9090)),
            max_request_bytes: DEFAULT_MAX_REQUEST_BYTES,
            drain_timeout_seconds: DEFAULT_DRAIN_TIMEOUT_SECONDS,
            force_close_timeout_seconds: DEFAULT_FORCE_CLOSE_TIMEOUT_SECONDS,
            allowed_origins: Vec::new(),
            tokens: Vec::new(),
            aliases: Vec::new(),
        }
    }
}

impl GatewayServerConfig {
    /// Parse a TOML file without reading secrets or loading endpoints.
    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self = toml::from_str(&source)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration without environment, network, or model side effects.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.public_bind == self.management_bind {
            bail!("public_bind and management_bind must be different");
        }
        if self.max_request_bytes == 0 {
            bail!("max_request_bytes must be greater than zero");
        }
        if self.drain_timeout_seconds == 0 {
            bail!("drain_timeout_seconds must be greater than zero");
        }
        if self.force_close_timeout_seconds == 0 {
            bail!("force_close_timeout_seconds must be greater than zero");
        }
        if self.tokens.is_empty() {
            bail!("at least one scoped bearer token is required");
        }
        if self.aliases.is_empty() {
            bail!("at least one alias is required");
        }

        let mut aliases = BTreeSet::new();
        for alias in &self.aliases {
            validate_name(&alias.name, "alias name")?;
            if !aliases.insert(alias.name.as_str()) {
                bail!("duplicate alias: {}", alias.name);
            }
            alias.validate()?;
        }
        let mut callers = BTreeSet::new();
        for token in &self.tokens {
            validate_env_name(&token.env)?;
            validate_name(&token.caller, "token caller")?;
            if !callers.insert(token.caller.as_str()) {
                bail!("duplicate token caller: {}", token.caller);
            }
            token.validate(&aliases)?;
        }
        for origin in &self.allowed_origins {
            if origin.is_empty() || origin.trim() != origin {
                bail!("allowed origins must be non-empty exact values");
            }
        }
        Ok(())
    }

    /// Load every configured endpoint and build the immutable adapter.
    pub async fn build_adapter(&self) -> anyhow::Result<GatewayAdapter> {
        let mut client = CogentClient::new();
        let mut endpoints = HashMap::new();
        for alias in &self.aliases {
            let descriptor = alias.endpoint.descriptor()?;
            let endpoint = client
                .add(alias.name.clone(), descriptor)
                .await
                .with_context(|| format!("failed to load alias {}", alias.name))?;
            endpoints.insert(alias.name.clone(), endpoint);
        }

        let mut builder = GatewayAdapter::builder(CogentClientExecutor::new(client));
        for alias in &self.aliases {
            let endpoint = endpoints
                .remove(&alias.name)
                .context("loaded endpoint disappeared")?;
            builder = builder
                .alias(GatewayAlias::new(
                    alias.name.clone(),
                    endpoint,
                    alias.operations.operation_set(),
                    alias.limits.gateway_limits(),
                )?)
                .with_context(|| format!("failed to register alias {}", alias.name))?;
        }
        builder.build().map_err(Into::into)
    }

    /// Load bearer secrets and access scopes from the environment.
    pub fn load_tokens(&self) -> anyhow::Result<Vec<LoadedToken>> {
        self.tokens
            .iter()
            .map(|token| {
                let secret = required_env(&token.env)?;
                if secret.chars().any(char::is_whitespace) {
                    bail!("{} must not contain whitespace", token.env);
                }
                let access = if token.access.is_empty() {
                    GatewayAccess::all()
                } else {
                    GatewayAccess::new(
                        token
                            .access
                            .iter()
                            .map(|scope| (scope.alias.clone(), scope.operations.operation_set())),
                    )?
                };
                Ok(LoadedToken {
                    secret,
                    caller: GatewayCaller {
                        id: Some(token.caller.clone()),
                        access,
                    },
                })
            })
            .collect()
    }

    /// Graceful drain timeout.
    pub fn drain_timeout(&self) -> Duration {
        Duration::from_secs(self.drain_timeout_seconds)
    }

    /// Forced close timeout after cancellation.
    pub fn force_close_timeout(&self) -> Duration {
        Duration::from_secs(self.force_close_timeout_seconds)
    }
}

/// Scoped bearer token configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenConfig {
    /// Environment variable containing the bearer secret.
    pub env: String,
    /// Stable caller label used for limits and logs.
    pub caller: String,
    /// Empty means all aliases and operations.
    #[serde(default)]
    pub access: Vec<AccessConfig>,
}

impl TokenConfig {
    fn validate(&self, aliases: &BTreeSet<&str>) -> anyhow::Result<()> {
        let mut seen = BTreeSet::new();
        for scope in &self.access {
            if !aliases.contains(scope.alias.as_str()) {
                bail!("token access references unknown alias: {}", scope.alias);
            }
            if !seen.insert(scope.alias.as_str()) {
                bail!("duplicate token access alias: {}", scope.alias);
            }
            scope.operations.validate()?;
        }
        Ok(())
    }
}

/// One alias access scope.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccessConfig {
    /// Public alias.
    pub alias: String,
    /// Allowed operations.
    pub operations: OperationsConfig,
}

/// Alias configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct AliasConfig {
    /// Public alias name and private client endpoint ID.
    pub name: String,
    /// Exposed operations.
    #[serde(default)]
    pub operations: OperationsConfig,
    /// Replica-local request limits.
    #[serde(default)]
    pub limits: AliasLimitsConfig,
    /// Endpoint loaded for the alias.
    #[serde(flatten)]
    pub endpoint: EndpointConfig,
}

impl AliasConfig {
    fn validate(&self) -> anyhow::Result<()> {
        self.operations.validate()?;
        self.limits.validate()?;
        self.endpoint.validate()
    }
}

/// Operation switches.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OperationsConfig {
    /// Enable query.
    pub query: bool,
    /// Enable chat.
    pub chat: bool,
    /// Enable embeddings.
    pub embed: bool,
}

impl OperationsConfig {
    fn validate(&self) -> anyhow::Result<()> {
        if !self.query && !self.chat && !self.embed {
            bail!("operation set must enable at least one operation");
        }
        Ok(())
    }

    fn operation_set(&self) -> OperationSet {
        let mut operations = Vec::new();
        if self.query {
            operations.push(Operation::Query);
        }
        if self.chat {
            operations.push(Operation::Chat);
        }
        if self.embed {
            operations.push(Operation::Embed);
        }
        OperationSet::new(operations)
    }
}

impl Default for OperationsConfig {
    fn default() -> Self {
        Self {
            query: true,
            chat: true,
            embed: true,
        }
    }
}

/// Alias limit configuration.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AliasLimitsConfig {
    /// Shared alias limits.
    pub global: RequestLimitsConfig,
    /// Per-caller limits.
    pub per_caller: Option<RequestLimitsConfig>,
    /// Maximum caller IDs retained for per-caller state.
    pub max_tracked_callers: Option<usize>,
}

impl AliasLimitsConfig {
    fn validate(&self) -> anyhow::Result<()> {
        self.global.validate()?;
        if let Some(per_caller) = &self.per_caller {
            per_caller.validate()?;
            if self.max_tracked_callers == Some(0) {
                bail!("max_tracked_callers must be greater than zero");
            }
        }
        Ok(())
    }

    fn gateway_limits(&self) -> GatewayAliasLimits {
        GatewayAliasLimits {
            global: self.global.gateway_limits(),
            per_caller: self
                .per_caller
                .as_ref()
                .map(RequestLimitsConfig::gateway_limits),
            max_tracked_callers: self
                .max_tracked_callers
                .unwrap_or(cogentlm_gateway::DEFAULT_MAX_TRACKED_CALLERS),
        }
    }
}

/// Request limit values.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RequestLimitsConfig {
    /// Maximum concurrent requests.
    pub max_concurrent_requests: Option<usize>,
    /// Maximum requests per rolling minute.
    pub max_requests_per_minute: Option<u32>,
    /// Maximum requests since startup.
    pub max_requests_total: Option<u64>,
}

impl RequestLimitsConfig {
    fn validate(&self) -> anyhow::Result<()> {
        if self.max_concurrent_requests == Some(0)
            || self.max_requests_per_minute == Some(0)
            || self.max_requests_total == Some(0)
        {
            bail!("request limits must be greater than zero");
        }
        Ok(())
    }

    fn gateway_limits(&self) -> GatewayRequestLimits {
        GatewayRequestLimits {
            max_concurrent_requests: self.max_concurrent_requests,
            max_requests_per_minute: self.max_requests_per_minute,
            max_requests_total: self.max_requests_total,
        }
    }
}

/// Endpoint variants supported by the standalone service.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum EndpointConfig {
    /// In-process GGUF model.
    Local {
        /// Model artifact path.
        model: PathBuf,
        /// Native runtime settings.
        #[serde(default)]
        runtime: NativeRuntimeConfig,
    },
    /// OpenAI endpoint.
    Openai {
        /// Provider model name.
        model: String,
        /// API key environment variable.
        api_key_env: String,
        /// Optional alternate base URL.
        base_url: Option<String>,
        /// Optional total timeout for unary requests and idle stream timeout.
        timeout_seconds: Option<u64>,
    },
    /// OpenAI-compatible endpoint.
    OpenaiCompatible {
        /// Provider model name.
        model: String,
        /// Provider base URL.
        base_url: String,
        /// Bearer token environment variable.
        token_env: String,
        /// Optional correlation header used by the provider.
        correlation_header: Option<String>,
        /// Optional timeout.
        timeout_seconds: Option<u64>,
    },
    /// Anthropic endpoint.
    Anthropic {
        /// Provider model name.
        model: String,
        /// API key environment variable.
        api_key_env: String,
        /// Optional alternate base URL.
        base_url: Option<String>,
        /// Optional API version.
        version: Option<String>,
        /// Optional timeout.
        timeout_seconds: Option<u64>,
    },
}

impl EndpointConfig {
    fn validate(&self) -> anyhow::Result<()> {
        match self {
            Self::Local { model, .. } => {
                if model.as_os_str().is_empty() {
                    bail!("local model path must not be empty");
                }
            }
            Self::Openai {
                model,
                api_key_env,
                timeout_seconds,
                ..
            }
            | Self::Anthropic {
                model,
                api_key_env,
                timeout_seconds,
                ..
            } => {
                validate_name(model, "provider model")?;
                validate_env_name(api_key_env)?;
                validate_timeout(*timeout_seconds)?;
            }
            Self::OpenaiCompatible {
                model,
                base_url,
                token_env,
                correlation_header,
                timeout_seconds,
            } => {
                validate_name(model, "provider model")?;
                validate_name(base_url, "provider base URL")?;
                validate_env_name(token_env)?;
                if let Some(header) = correlation_header {
                    validate_name(header, "correlation header")?;
                }
                validate_timeout(*timeout_seconds)?;
            }
        }
        Ok(())
    }

    fn descriptor(&self) -> anyhow::Result<EndpointDescriptor> {
        match self {
            Self::Local { model, runtime } => {
                Ok(EndpointDescriptor::local(model.clone(), runtime.clone()))
            }
            Self::Openai {
                model,
                api_key_env,
                base_url,
                timeout_seconds,
            } => Ok(EndpointDescriptor::provider(
                ProviderEndpointConfig::OpenAi(OpenAiProviderConfig {
                    model: model.clone(),
                    api_key: ProviderSecret::new(required_env(api_key_env)?),
                    base_url: base_url.clone(),
                    timeout: timeout(*timeout_seconds),
                }),
            )),
            Self::OpenaiCompatible {
                model,
                base_url,
                token_env,
                correlation_header,
                timeout_seconds,
            } => Ok(EndpointDescriptor::provider(
                ProviderEndpointConfig::OpenAiCompatible(OpenAiCompatibleProviderConfig {
                    model: model.clone(),
                    base_url: base_url.clone(),
                    auth: ProviderAuthConfig::Bearer(ProviderSecret::new(required_env(token_env)?)),
                    static_headers: Vec::new(),
                    correlation_header: correlation_header.clone(),
                    timeout: timeout(*timeout_seconds),
                }),
            )),
            Self::Anthropic {
                model,
                api_key_env,
                base_url,
                version,
                timeout_seconds,
            } => Ok(EndpointDescriptor::provider(
                ProviderEndpointConfig::Anthropic(AnthropicProviderConfig {
                    model: model.clone(),
                    api_key: ProviderSecret::new(required_env(api_key_env)?),
                    base_url: base_url.clone(),
                    version: version.clone(),
                    timeout: timeout(*timeout_seconds),
                }),
            )),
        }
    }
}

/// Loaded bearer token.
#[derive(Clone)]
pub struct LoadedToken {
    pub(crate) secret: String,
    pub(crate) caller: GatewayCaller,
}

impl LoadedToken {
    /// Create a loaded token for embedding the HTTP service or tests.
    pub fn new(secret: impl Into<String>, caller: GatewayCaller) -> anyhow::Result<Self> {
        let secret = secret.into();
        if secret.is_empty() || secret.chars().any(char::is_whitespace) {
            bail!("bearer secret must be non-empty and contain no whitespace");
        }
        Ok(Self { secret, caller })
    }
}

fn required_env(name: &str) -> anyhow::Result<String> {
    let value = std::env::var(name).with_context(|| format!("{name} is required"))?;
    if value.trim().is_empty() {
        bail!("{name} must not be empty");
    }
    Ok(value)
}

fn validate_env_name(name: &str) -> anyhow::Result<()> {
    validate_name(name, "environment variable")?;
    if !name
        .bytes()
        .all(|byte| byte == b'_' || byte.is_ascii_alphanumeric())
    {
        bail!("invalid environment variable name: {name}");
    }
    Ok(())
}

fn validate_name(value: &str, field: &str) -> anyhow::Result<()> {
    if value.is_empty() || value.trim() != value {
        bail!("{field} must be a non-empty trimmed value");
    }
    Ok(())
}

fn validate_timeout(value: Option<u64>) -> anyhow::Result<()> {
    if value == Some(0) {
        bail!("timeout_seconds must be greater than zero");
    }
    Ok(())
}

fn timeout(seconds: Option<u64>) -> Option<Duration> {
    seconds.map(Duration::from_secs)
}
