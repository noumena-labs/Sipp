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
use cogentlm_gateway::GatewayRoutes;
use serde::Deserialize;

/// Standalone application configuration.
#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GatewayServerConfig {
    /// Public inference listener.
    pub public_bind: SocketAddr,
    /// Management listener.
    pub management_bind: SocketAddr,
    /// Maximum request body bytes.
    pub max_request_bytes: usize,
    /// Browser origins accepted by the public listener.
    pub allowed_origins: Vec<String>,
    /// Application-owned route selection.
    pub routes: RouteConfig,
    /// Bearer tokens and target access loaded from environment variables.
    pub tokens: Vec<TokenConfig>,
    /// Public targets backed by local or provider endpoints.
    pub targets: Vec<TargetConfig>,
    /// Application-wide concurrent request limit.
    pub max_concurrent_requests: Option<usize>,
}

impl Default for GatewayServerConfig {
    fn default() -> Self {
        Self {
            public_bind: SocketAddr::from(([0, 0, 0, 0], 8080)),
            management_bind: SocketAddr::from(([0, 0, 0, 0], 9090)),
            max_request_bytes: 1 << 20,
            allowed_origins: Vec::new(),
            routes: RouteConfig::default(),
            tokens: Vec::new(),
            targets: Vec::new(),
            max_concurrent_requests: None,
        }
    }
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
        for target in &self.targets {
            let endpoint = client
                .add(target.name.clone(), target.endpoint.descriptor()?)
                .await
                .with_context(|| format!("failed to load target {}", target.name))?;
            targets.insert(target.name.clone(), endpoint);
        }
        Ok(GatewayServerRuntime {
            client: Arc::new(client),
            targets: Arc::new(targets),
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

    /// Convert application route configuration to toolkit routes.
    pub fn gateway_routes(&self) -> GatewayRoutes {
        self.routes.clone().into()
    }
}

/// Loaded application runtime used by explicit route handlers.
#[derive(Clone)]
pub struct GatewayServerRuntime {
    pub(crate) client: Arc<CogentClient>,
    pub(crate) targets: Arc<BTreeMap<String, EndpointRef>>,
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
        ])?;
        Ok(())
    }
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

/// Endpoint variants selected by this first-party application.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum EndpointConfig {
    Local {
        model: PathBuf,
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
