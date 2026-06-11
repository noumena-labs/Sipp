use std::collections::{BTreeMap, BTreeSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{bail, Context};
use cogentlm_client::{
    AnthropicProviderConfig, CogentClient, EndpointDescriptor, EndpointRef,
    OpenAiCompatibleProviderConfig, OpenAiProviderConfig, ProviderAuthConfig,
    ProviderEndpointConfig, ProviderSecret,
};
use cogentlm_engine::engine::NativeRuntimeConfig;
#[cfg(test)]
use cogentlm_engine::lifecycle::BackendCapabilities;
use cogentlm_engine::lifecycle::{
    BackendPlan, BackendPolicy, BackendPreference, BackendSelection, ModelLoadOptions, StatsMode,
};
use cogentlm_gateway::GatewayRoutes;
use serde::Deserialize;

/// Standalone application configuration.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GatewayServerConfig {
    /// Public inference listener.
    #[serde(default = "default_public_bind")]
    pub public_bind: SocketAddr,
    /// Management listener.
    #[serde(default = "default_management_bind")]
    pub management_bind: SocketAddr,
    /// Maximum request body bytes.
    #[serde(default = "default_max_request_bytes")]
    pub max_request_bytes: usize,
    /// Browser origins accepted by the public listener.
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    /// Environment variable containing the Admin Dashboard password.
    #[serde(default)]
    pub admin_password_env: String,
    /// Application-owned route selection.
    #[serde(default = "default_routes")]
    pub routes: RouteConfig,
    /// Bearer tokens and target access loaded from environment variables.
    #[serde(default)]
    pub tokens: Vec<TokenConfig>,
    /// Public targets backed by local or provider endpoints.
    #[serde(default)]
    pub targets: Vec<TargetConfig>,
    /// Application-wide concurrent request limit.
    #[serde(default)]
    pub max_concurrent_requests: Option<usize>,
    /// In-memory security and client identification settings.
    pub security: SecurityConfig,
}

fn default_public_bind() -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], 8080))
}

fn default_management_bind() -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], 9090))
}

const fn default_max_request_bytes() -> usize {
    1 << 20
}

fn default_routes() -> RouteConfig {
    RouteConfig::default()
}

impl GatewayServerConfig {
    /// Parse and validate a TOML configuration file.
    pub fn from_path(path: &Path) -> anyhow::Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        let config: Self = toml::from_str(&source)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        config.validate()?;
        Ok(config)
    }

    /// Validate configuration without loading secrets or endpoints.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.public_bind == self.management_bind {
            bail!("public_bind and management_bind must be different");
        }
        if self.max_request_bytes == 0 {
            bail!("max_request_bytes must be greater than zero");
        }
        if self.max_concurrent_requests == Some(0) {
            bail!("max_concurrent_requests must be greater than zero");
        }
        self.security.validate()?;
        if self.admin_password_env.trim().is_empty() {
            bail!("admin_password_env must not be empty");
        }
        validate_env_name(&self.admin_password_env)?;
        self.routes.validate()?;
        if self.tokens.is_empty() {
            bail!("at least one bearer token is required");
        }
        if self.targets.is_empty() {
            bail!("at least one target is required");
        }

        let mut targets = BTreeSet::new();
        for target in &self.targets {
            validate_name(&target.name, "target name")?;
            if !targets.insert(target.name.as_str()) {
                bail!("duplicate target: {}", target.name);
            }
            target.endpoint.validate()?;
        }
        let mut callers = BTreeSet::new();
        for token in &self.tokens {
            validate_env_name(&token.env)?;
            validate_name(&token.caller, "token caller")?;
            if !callers.insert(token.caller.as_str()) {
                bail!("duplicate token caller: {}", token.caller);
            }
            for target in &token.targets {
                if !targets.contains(target.as_str()) {
                    bail!("token references unknown target: {target}");
                }
            }
        }
        Ok(())
    }

    /// Load endpoints and return the application-owned client runtime.
    pub async fn build_runtime(&self) -> anyhow::Result<GatewayServerRuntime> {
        let mut client = CogentClient::new();
        let mut targets = BTreeMap::new();
        let mut summaries = Vec::new();
        for target in &self.targets {
            let (descriptor, summary) = target.endpoint.descriptor_and_summary(&target.name)?;
            let endpoint = client
                .add(target.name.clone(), descriptor)
                .await
                .with_context(|| format!("failed to load target {}", target.name))?;
            targets.insert(target.name.clone(), endpoint);
            summaries.push(summary);
        }
        Ok(GatewayServerRuntime {
            client: Arc::new(client),
            targets: Arc::new(targets),
            target_summaries: Arc::new(summaries),
        })
    }

    /// Load application bearer token policy.
    pub fn load_tokens(&self) -> anyhow::Result<Vec<LoadedToken>> {
        self.tokens
            .iter()
            .map(|token| {
                let secret = required_env(&token.env)?;
                if secret.chars().any(char::is_whitespace) {
                    bail!("{} must not contain whitespace", token.env);
                }
                Ok(LoadedToken {
                    secret,
                    caller: token.caller.clone(),
                    targets: token.targets.iter().cloned().collect(),
                })
            })
            .collect()
    }

    /// Load the Admin Dashboard password from its configured environment variable.
    pub fn load_admin_password(&self) -> anyhow::Result<String> {
        required_env(&self.admin_password_env)
    }

    /// Convert application route configuration to toolkit routes.
    pub fn gateway_routes(&self) -> GatewayRoutes {
        self.routes.clone().into()
    }
}

/// Runtime security configuration for the standalone gateway.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SecurityConfig {
    /// Client IP extraction settings.
    pub client_ip: ClientIpConfig,
    /// In-memory per-client rate limiting settings.
    pub rate_limit: RateLimitConfig,
}

impl SecurityConfig {
    fn validate(&self) -> anyhow::Result<()> {
        self.client_ip.validate()?;
        self.rate_limit.validate()
    }
}

/// Client IP extraction settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClientIpConfig {
    /// Source used for the client address.
    pub source: ClientIpSource,
    /// Trusted proxy CIDRs allowed to supply forwarding headers.
    pub trusted_proxy_cidrs: Vec<String>,
}

impl ClientIpConfig {
    fn validate(&self) -> anyhow::Result<()> {
        for cidr in &self.trusted_proxy_cidrs {
            validate_ip_cidr(cidr)?;
        }
        Ok(())
    }
}

/// Source used to identify public clients for in-memory controls.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ClientIpSource {
    /// Use the TCP peer address.
    Peer,
    /// Use the leftmost `X-Forwarded-For` value from trusted proxies.
    XForwardedFor,
    /// Use `X-Real-IP` from trusted proxies.
    XRealIp,
}

impl ClientIpSource {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Peer => "peer",
            Self::XForwardedFor => "x_forwarded_for",
            Self::XRealIp => "x_real_ip",
        }
    }
}

/// In-memory per-client rate limiting settings.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RateLimitConfig {
    /// Whether per-client rate limiting is enabled.
    pub enabled: bool,
    /// Token refill rate in requests per minute.
    pub requests_per_minute: u32,
    /// Token bucket capacity.
    pub burst: u32,
}

impl RateLimitConfig {
    fn validate(&self) -> anyhow::Result<()> {
        if self.requests_per_minute == 0 {
            bail!("security.rate_limit.requests_per_minute must be greater than zero");
        }
        if self.burst == 0 {
            bail!("security.rate_limit.burst must be greater than zero");
        }
        Ok(())
    }
}

/// Loaded application runtime used by explicit route handlers.
#[derive(Clone)]
pub struct GatewayServerRuntime {
    pub(crate) client: Arc<CogentClient>,
    pub(crate) targets: Arc<BTreeMap<String, EndpointRef>>,
    pub(crate) target_summaries: Arc<Vec<TargetSummary>>,
}

/// TOML-selectable routes.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RouteConfig {
    pub query: String,
    pub chat: String,
    pub embed: String,
    pub index: Option<String>,
    pub health: Option<String>,
    pub readiness: Option<String>,
    pub metrics: Option<String>,
    #[serde(default = "default_admin_route")]
    pub admin: Option<String>,
}

impl Default for RouteConfig {
    fn default() -> Self {
        GatewayRoutes::default().into()
    }
}

impl RouteConfig {
    fn validate(&self) -> anyhow::Result<()> {
        for (name, route) in [
            ("query", Some(self.query.as_str())),
            ("chat", Some(self.chat.as_str())),
            ("embed", Some(self.embed.as_str())),
            ("index", self.index.as_deref()),
            ("health", self.health.as_deref()),
            ("readiness", self.readiness.as_deref()),
            ("metrics", self.metrics.as_deref()),
            ("admin", self.admin.as_deref()),
        ] {
            if let Some(route) = route {
                if !route.starts_with('/') || route.contains('?') || route.contains('#') {
                    bail!("{name} route must be an absolute path");
                }
            }
        }
        reject_duplicate_routes([
            ("query", Some(self.query.as_str())),
            ("chat", Some(self.chat.as_str())),
            ("embed", Some(self.embed.as_str())),
        ])?;
        reject_duplicate_routes([
            ("index", self.index.as_deref()),
            ("health", self.health.as_deref()),
            ("readiness", self.readiness.as_deref()),
            ("metrics", self.metrics.as_deref()),
            ("admin", self.admin.as_deref()),
        ])?;
        Ok(())
    }
}

fn default_admin_route() -> Option<String> {
    Some("/admin".to_string())
}

fn reject_duplicate_routes<const N: usize>(
    routes: [(&str, Option<&str>); N],
) -> anyhow::Result<()> {
    let mut seen = BTreeMap::new();
    for (name, route) in routes {
        let Some(route) = route else {
            continue;
        };
        if let Some(previous) = seen.insert(route, name) {
            bail!("{name} route duplicates {previous} route: {route}");
        }
    }
    Ok(())
}

impl From<RouteConfig> for GatewayRoutes {
    fn from(routes: RouteConfig) -> Self {
        Self {
            query: routes.query,
            chat: routes.chat,
            embed: routes.embed,
            index: routes.index,
            health: routes.health,
            readiness: routes.readiness,
            metrics: routes.metrics,
        }
    }
}

impl From<GatewayRoutes> for RouteConfig {
    fn from(routes: GatewayRoutes) -> Self {
        Self {
            query: routes.query,
            chat: routes.chat,
            embed: routes.embed,
            index: routes.index,
            health: routes.health,
            readiness: routes.readiness,
            metrics: routes.metrics,
            admin: default_admin_route(),
        }
    }
}

/// Bearer token policy.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TokenConfig {
    /// Environment variable containing the secret.
    pub env: String,
    /// Stable caller label.
    pub caller: String,
    /// Allowed targets. Empty grants every configured target.
    #[serde(default)]
    pub targets: Vec<String>,
}

/// Public target and endpoint implementation.
#[derive(Debug, Clone, Deserialize)]
pub struct TargetConfig {
    pub name: String,
    #[serde(flatten)]
    pub endpoint: EndpointConfig,
}

/// Dashboard-safe target metadata.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TargetSummary {
    pub(crate) name: String,
    pub(crate) kind: TargetKind,
    pub(crate) model: String,
    pub(crate) backend: Option<BackendSelection>,
    pub(crate) provider_base_url: Option<String>,
}

/// Target implementation family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TargetKind {
    Local,
    OpenAi,
    OpenAiCompatible,
    Anthropic,
}

impl TargetKind {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::OpenAi => "openai",
            Self::OpenAiCompatible => "openai_compatible",
            Self::Anthropic => "anthropic",
        }
    }
}

/// Gateway-supported local inference backend selection.
#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GatewayBackendPreference {
    #[default]
    Auto,
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

impl GatewayBackendPreference {
    pub(crate) const fn as_engine(self) -> BackendPreference {
        match self {
            Self::Auto => BackendPreference::Auto,
            Self::Cpu => BackendPreference::Cpu,
            Self::Cuda => BackendPreference::Cuda,
            Self::Metal => BackendPreference::Metal,
            Self::Vulkan => BackendPreference::Vulkan,
        }
    }
}

/// Endpoint variants selected by this first-party application.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum EndpointConfig {
    Local {
        model: PathBuf,
        #[serde(default)]
        backend: GatewayBackendPreference,
        #[serde(default)]
        stats: StatsMode,
        #[serde(default)]
        runtime: NativeRuntimeConfig,
    },
    Openai {
        model: String,
        api_key_env: String,
        base_url: Option<String>,
        timeout_seconds: Option<u64>,
    },
    OpenaiCompatible {
        model: String,
        base_url: String,
        token_env: String,
        correlation_header: Option<String>,
        timeout_seconds: Option<u64>,
    },
    Anthropic {
        model: String,
        api_key_env: String,
        base_url: Option<String>,
        version: Option<String>,
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

    fn descriptor_and_summary(
        &self,
        target_name: &str,
    ) -> anyhow::Result<(EndpointDescriptor, TargetSummary)> {
        match self {
            Self::Local {
                model,
                backend,
                stats,
                runtime,
            } => {
                let plan = local_backend_plan(*backend, *stats, runtime.clone())?;
                Ok((
                    EndpointDescriptor::local(model.clone(), plan.config),
                    TargetSummary {
                        name: target_name.to_string(),
                        kind: TargetKind::Local,
                        model: display_model_name(model),
                        backend: Some(plan.selection),
                        provider_base_url: None,
                    },
                ))
            }
            Self::Openai {
                model,
                api_key_env,
                base_url,
                timeout_seconds,
            } => Ok((
                EndpointDescriptor::provider(ProviderEndpointConfig::OpenAi(
                    OpenAiProviderConfig {
                        model: model.clone(),
                        api_key: ProviderSecret::new(required_env(api_key_env)?),
                        base_url: base_url.clone(),
                        timeout: timeout(*timeout_seconds),
                    },
                )),
                TargetSummary {
                    name: target_name.to_string(),
                    kind: TargetKind::OpenAi,
                    model: model.clone(),
                    backend: None,
                    provider_base_url: base_url.clone(),
                },
            )),
            Self::OpenaiCompatible {
                model,
                base_url,
                token_env,
                correlation_header,
                timeout_seconds,
            } => Ok((
                EndpointDescriptor::provider(ProviderEndpointConfig::OpenAiCompatible(
                    OpenAiCompatibleProviderConfig {
                        model: model.clone(),
                        base_url: base_url.clone(),
                        auth: ProviderAuthConfig::Bearer(ProviderSecret::new(required_env(
                            token_env,
                        )?)),
                        static_headers: Vec::new(),
                        correlation_header: correlation_header.clone(),
                        timeout: timeout(*timeout_seconds),
                    },
                )),
                TargetSummary {
                    name: target_name.to_string(),
                    kind: TargetKind::OpenAiCompatible,
                    model: model.clone(),
                    backend: None,
                    provider_base_url: Some(base_url.clone()),
                },
            )),
            Self::Anthropic {
                model,
                api_key_env,
                base_url,
                version,
                timeout_seconds,
            } => Ok((
                EndpointDescriptor::provider(ProviderEndpointConfig::Anthropic(
                    AnthropicProviderConfig {
                        model: model.clone(),
                        api_key: ProviderSecret::new(required_env(api_key_env)?),
                        base_url: base_url.clone(),
                        version: version.clone(),
                        timeout: timeout(*timeout_seconds),
                    },
                )),
                TargetSummary {
                    name: target_name.to_string(),
                    kind: TargetKind::Anthropic,
                    model: model.clone(),
                    backend: None,
                    provider_base_url: base_url.clone(),
                },
            )),
        }
    }
}

fn local_backend_plan(
    backend: GatewayBackendPreference,
    stats: StatsMode,
    runtime: NativeRuntimeConfig,
) -> anyhow::Result<BackendPlan> {
    BackendPolicy::select(&ModelLoadOptions {
        backend: backend.as_engine(),
        stats,
        runtime,
    })
    .map_err(anyhow::Error::from)
}

#[cfg(test)]
pub(crate) fn local_backend_plan_with_capabilities(
    backend: GatewayBackendPreference,
    stats: StatsMode,
    runtime: NativeRuntimeConfig,
    capabilities: &BackendCapabilities,
) -> anyhow::Result<BackendPlan> {
    BackendPolicy::select_with_capabilities(
        &ModelLoadOptions {
            backend: backend.as_engine(),
            stats,
            runtime,
        },
        capabilities,
    )
    .map_err(anyhow::Error::from)
}

fn display_model_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model.gguf")
        .to_string()
}

/// Loaded application authentication policy.
#[derive(Clone)]
pub struct LoadedToken {
    pub(crate) secret: String,
    pub(crate) caller: String,
    pub(crate) targets: BTreeSet<String>,
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

fn validate_trimmed(value: &str, field: &str) -> anyhow::Result<()> {
    if value.is_empty() || value.trim() != value {
        bail!("{field} must be a non-empty trimmed value");
    }
    Ok(())
}

fn validate_name(value: &str, field: &str) -> anyhow::Result<()> {
    validate_trimmed(value, field)
}

fn validate_timeout(value: Option<u64>) -> anyhow::Result<()> {
    if value == Some(0) {
        bail!("timeout_seconds must be greater than zero");
    }
    Ok(())
}

fn validate_ip_cidr(value: &str) -> anyhow::Result<()> {
    let (address, prefix) = value
        .split_once('/')
        .with_context(|| format!("trusted proxy CIDR must include a prefix: {value}"))?;
    let address = address
        .parse::<std::net::IpAddr>()
        .with_context(|| format!("invalid trusted proxy CIDR address: {value}"))?;
    let prefix = prefix
        .parse::<u8>()
        .with_context(|| format!("invalid trusted proxy CIDR prefix: {value}"))?;
    let max_prefix = match address {
        std::net::IpAddr::V4(_) => 32,
        std::net::IpAddr::V6(_) => 128,
    };
    if prefix > max_prefix {
        bail!("trusted proxy CIDR prefix is too large: {value}");
    }
    Ok(())
}

fn timeout(seconds: Option<u64>) -> Option<Duration> {
    seconds.map(Duration::from_secs)
}
