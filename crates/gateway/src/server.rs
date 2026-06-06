use std::{
    collections::BTreeMap,
    fmt,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use futures_util::StreamExt;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{
    protocol::{validate_gateway_options, validate_non_empty, validate_text_options},
    BackendChatRequest, BackendEmbedRequest, BackendEmbeddingOutput, BackendQueryRequest,
    BackendTextOutput, GatewayBackend, GatewayError, GatewayErrorKind, GatewayResult,
    GatewayStream, GatewayStreamEvent, Operation, OperationSet,
};

/// Maximum distinct caller IDs tracked per alias by default.
pub const DEFAULT_MAX_TRACKED_CALLERS: usize = 10_000;

/// Rate, quota, and concurrency limits for one policy scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GatewayRequestLimits {
    /// Maximum concurrent requests for the scope.
    pub max_concurrent_requests: Option<usize>,
    /// Maximum requests per rolling minute for the scope.
    pub max_requests_per_minute: Option<u32>,
    /// Maximum total requests allowed since adapter startup.
    pub max_requests_total: Option<u64>,
}

/// Alias-specific gateway policy limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayAliasLimits {
    /// Limits that apply to the alias across every caller.
    pub global: GatewayRequestLimits,
    /// Limits that apply independently to each caller ID.
    pub per_caller: Option<GatewayRequestLimits>,
    /// Maximum number of caller IDs tracked for this alias.
    pub max_tracked_callers: usize,
}

impl Default for GatewayAliasLimits {
    fn default() -> Self {
        Self {
            global: GatewayRequestLimits::default(),
            per_caller: None,
            max_tracked_callers: DEFAULT_MAX_TRACKED_CALLERS,
        }
    }
}

/// Public gateway alias and its server-side backend.
#[derive(Clone)]
pub struct GatewayAlias {
    name: String,
    operations: OperationSet,
    backend: Arc<dyn GatewayBackend>,
    limits: GatewayAliasLimits,
    global_concurrency: Option<Arc<Semaphore>>,
}

impl GatewayAlias {
    /// Create a public alias backed by a server-side backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the alias name, operation set, or limits are
    /// invalid.
    pub fn new(
        name: impl Into<String>,
        operations: OperationSet,
        backend: Arc<dyn GatewayBackend>,
        limits: GatewayAliasLimits,
    ) -> GatewayResult<Self> {
        let name = name.into();
        validate_gateway_alias_name(&name)?;
        if operations.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway alias operations must not be empty",
            ));
        }
        validate_alias_limits(limits)?;
        let global_concurrency = limits
            .global
            .max_concurrent_requests
            .map(|limit| Arc::new(Semaphore::new(limit)));
        Ok(Self {
            name,
            operations,
            backend,
            limits,
            global_concurrency,
        })
    }

    /// Return the public alias name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return enabled operations for this alias.
    pub fn operations(&self) -> &OperationSet {
        &self.operations
    }

    /// Return configured policy limits for this alias.
    pub fn limits(&self) -> GatewayAliasLimits {
        self.limits
    }
}

/// Access scope granted to a caller by the host application.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayAccess {
    aliases: Option<BTreeMap<String, OperationSet>>,
}

impl GatewayAccess {
    /// Allow every alias and operation.
    pub const fn all() -> Self {
        Self { aliases: None }
    }

    /// Restrict access to the listed aliases and operations.
    ///
    /// # Errors
    ///
    /// Returns an error when aliases are blank, duplicated, or mapped to no
    /// operations.
    pub fn new(aliases: impl IntoIterator<Item = (String, OperationSet)>) -> GatewayResult<Self> {
        let mut access = BTreeMap::new();
        for (alias, operations) in aliases {
            validate_gateway_access_alias(&alias)?;
            if operations.is_empty() {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    "caller access operations must not be empty",
                ));
            }
            if access.insert(alias, operations).is_some() {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    "caller access aliases must not contain duplicates",
                ));
            }
        }
        if access.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "caller access aliases must not be empty",
            ));
        }
        Ok(Self {
            aliases: Some(access),
        })
    }

    /// Return whether this access scope allows the alias operation.
    pub fn allows(&self, alias: &str, operation: Operation) -> bool {
        match &self.aliases {
            None => true,
            Some(aliases) => aliases
                .get(alias)
                .is_some_and(|operations| operations.supports(operation)),
        }
    }
}

/// Authenticated caller context supplied by a host server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayCaller {
    /// Stable caller ID used for per-caller limits and observability.
    pub id: Option<String>,
    /// Alias and operation access granted to the caller.
    pub access: GatewayAccess,
}

impl GatewayCaller {
    /// Create an anonymous caller with full access.
    pub const fn anonymous() -> Self {
        Self {
            id: None,
            access: GatewayAccess::all(),
        }
    }

    /// Create a caller with a stable ID and full access.
    pub fn identified(id: impl Into<String>) -> GatewayResult<Self> {
        let id = id.into();
        validate_gateway_caller_id(&id)?;
        Ok(Self {
            id: Some(id),
            access: GatewayAccess::all(),
        })
    }
}

/// Framework-agnostic gateway adapter.
#[derive(Clone)]
pub struct GatewayAdapter {
    aliases: BTreeMap<String, GatewayAlias>,
    policy: Arc<Mutex<GatewayPolicyState>>,
}

impl GatewayAdapter {
    /// Create an empty gateway adapter.
    pub fn new() -> Self {
        Self {
            aliases: BTreeMap::new(),
            policy: Arc::new(Mutex::new(GatewayPolicyState::default())),
        }
    }

    /// Create a builder for a gateway adapter.
    pub fn builder() -> GatewayBuilder {
        GatewayBuilder::default()
    }

    /// Add an alias to the adapter.
    ///
    /// # Errors
    ///
    /// Returns an error when another alias with the same name already exists.
    pub fn add_alias(&mut self, alias: GatewayAlias) -> GatewayResult<()> {
        if self.aliases.contains_key(&alias.name) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!("duplicate gateway alias: {}", alias.name),
            ));
        }
        {
            let mut policy = self.policy_lock()?;
            if policy.aliases.contains_key(&alias.name) {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    format!("duplicate gateway alias policy: {}", alias.name),
                ));
            }
            policy.aliases.insert(
                alias.name.clone(),
                AliasPolicyState {
                    global: BucketState {
                        window_started: Instant::now(),
                        ..BucketState::default()
                    },
                    callers: BTreeMap::new(),
                },
            );
        }
        self.aliases.insert(alias.name.clone(), alias);
        Ok(())
    }

    /// Return an immutable view of configured aliases.
    pub fn aliases(&self) -> &BTreeMap<String, GatewayAlias> {
        &self.aliases
    }

    /// Run a raw-prompt query through an alias.
    ///
    /// # Errors
    ///
    /// Returns gateway errors for invalid requests, authorization failures,
    /// policy violations, or backend failures.
    pub async fn query(
        &self,
        caller: &GatewayCaller,
        alias: &str,
        request: BackendQueryRequest,
    ) -> GatewayResult<BackendTextOutput> {
        validate_gateway_options(&request.gateway_options)?;
        validate_non_empty(&request.prompt, "prompt")?;
        validate_text_options(&request.options)?;
        let alias = self.alias(caller, alias, Operation::Query)?;
        let permit = self.acquire(caller, alias)?;
        let backend = alias.backend.clone();
        let output = backend.query(request).await;
        drop(permit);
        output
    }

    /// Stream a raw-prompt query through an alias.
    ///
    /// # Errors
    ///
    /// Returns gateway errors for invalid requests, authorization failures,
    /// policy violations, or backend startup failures.
    pub async fn stream_query(
        &self,
        caller: &GatewayCaller,
        alias: &str,
        request: BackendQueryRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        validate_gateway_options(&request.gateway_options)?;
        validate_non_empty(&request.prompt, "prompt")?;
        validate_text_options(&request.options)?;
        let alias = self.alias(caller, alias, Operation::Query)?;
        let permit = self.acquire(caller, alias)?;
        let backend = alias.backend.clone();
        let stream = backend.stream_query(request).await?;
        Ok(hold_permit(stream, permit))
    }

    /// Run chat through an alias.
    ///
    /// # Errors
    ///
    /// Returns gateway errors for invalid requests, authorization failures,
    /// policy violations, or backend failures.
    pub async fn chat(
        &self,
        caller: &GatewayCaller,
        alias: &str,
        request: BackendChatRequest,
    ) -> GatewayResult<BackendTextOutput> {
        validate_gateway_options(&request.gateway_options)?;
        if request.messages.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "messages must not be empty",
            ));
        }
        validate_text_options(&request.options)?;
        let alias = self.alias(caller, alias, Operation::Chat)?;
        let permit = self.acquire(caller, alias)?;
        let backend = alias.backend.clone();
        let output = backend.chat(request).await;
        drop(permit);
        output
    }

    /// Stream chat through an alias.
    ///
    /// # Errors
    ///
    /// Returns gateway errors for invalid requests, authorization failures,
    /// policy violations, or backend startup failures.
    pub async fn stream_chat(
        &self,
        caller: &GatewayCaller,
        alias: &str,
        request: BackendChatRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        validate_gateway_options(&request.gateway_options)?;
        if request.messages.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "messages must not be empty",
            ));
        }
        validate_text_options(&request.options)?;
        let alias = self.alias(caller, alias, Operation::Chat)?;
        let permit = self.acquire(caller, alias)?;
        let backend = alias.backend.clone();
        let stream = backend.stream_chat(request).await?;
        Ok(hold_permit(stream, permit))
    }

    /// Run embedding through an alias.
    ///
    /// # Errors
    ///
    /// Returns gateway errors for invalid requests, authorization failures,
    /// policy violations, or backend failures.
    pub async fn embed(
        &self,
        caller: &GatewayCaller,
        alias: &str,
        request: BackendEmbedRequest,
    ) -> GatewayResult<BackendEmbeddingOutput> {
        validate_gateway_options(&request.gateway_options)?;
        validate_non_empty(&request.input, "input")?;
        let alias = self.alias(caller, alias, Operation::Embed)?;
        let permit = self.acquire(caller, alias)?;
        let backend = alias.backend.clone();
        let output = backend.embed(request).await;
        drop(permit);
        output
    }

    /// Return a redacted runtime snapshot for status UIs and diagnostics.
    ///
    /// # Errors
    ///
    /// Returns an internal error when policy state cannot be read.
    pub fn snapshot(&self) -> GatewayResult<GatewaySnapshot> {
        let policy = self.policy_lock()?;
        let aliases = self
            .aliases
            .iter()
            .map(|(name, alias)| {
                let state = policy.aliases.get(name);
                GatewayAliasSnapshot {
                    name: name.clone(),
                    query: alias.operations.supports(Operation::Query),
                    chat: alias.operations.supports(Operation::Chat),
                    embed: alias.operations.supports(Operation::Embed),
                    limits: alias.limits,
                    global_total_requests: state.map_or(0, |state| state.global.total_count),
                    global_window_requests: state.map_or(0, |state| state.global.window_count),
                    tracked_callers: state.map_or(0, |state| state.callers.len()),
                }
            })
            .collect();
        Ok(GatewaySnapshot { aliases })
    }

    fn alias(
        &self,
        caller: &GatewayCaller,
        model: &str,
        operation: Operation,
    ) -> GatewayResult<&GatewayAlias> {
        validate_non_empty(model, "model")?;
        if !caller.access.allows(model, operation) {
            return Err(GatewayError::new(
                GatewayErrorKind::Authorization,
                "caller is not allowed to use the requested model operation",
            ));
        }
        let alias = self.aliases.get(model).ok_or_else(|| {
            GatewayError::new(GatewayErrorKind::ModelNotFound, "model alias not found")
        })?;
        if !alias.operations.supports(operation) {
            return Err(GatewayError::new(
                GatewayErrorKind::UnsupportedFeature,
                format!("model alias does not support {}", operation.as_str()),
            ));
        }
        Ok(alias)
    }

    fn acquire(
        &self,
        caller: &GatewayCaller,
        alias: &GatewayAlias,
    ) -> GatewayResult<GatewayRequestPermit> {
        let caller_semaphore = self.caller_semaphore(caller, alias)?;
        let global_concurrency = match &alias.global_concurrency {
            Some(semaphore) => Some(semaphore.clone().try_acquire_owned().map_err(|_| {
                GatewayError::new(
                    GatewayErrorKind::Overloaded,
                    "model alias global concurrency limit exceeded",
                )
            })?),
            None => None,
        };
        let caller_concurrency = match caller_semaphore {
            Some(semaphore) => Some(semaphore.try_acquire_owned().map_err(|_| {
                GatewayError::new(
                    GatewayErrorKind::Overloaded,
                    "model alias caller concurrency limit exceeded",
                )
            })?),
            None => None,
        };

        let mut policy = self.policy_lock()?;
        let state = policy.aliases.get_mut(&alias.name).ok_or_else(|| {
            GatewayError::new(
                GatewayErrorKind::Internal,
                format!("missing policy state for alias: {}", alias.name),
            )
        })?;
        state
            .global
            .check_and_increment(alias.limits.global, "model alias global")?;
        if let Some(limits) = alias.limits.per_caller {
            let caller_id = caller_id(caller)?;
            let state = state.callers.get_mut(caller_id).ok_or_else(|| {
                GatewayError::new(
                    GatewayErrorKind::Internal,
                    "missing caller policy state after allocation",
                )
            })?;
            state
                .bucket
                .check_and_increment(limits, "model alias caller")?;
        }

        Ok(GatewayRequestPermit {
            _global_concurrency: global_concurrency,
            _caller_concurrency: caller_concurrency,
        })
    }

    fn caller_semaphore(
        &self,
        caller: &GatewayCaller,
        alias: &GatewayAlias,
    ) -> GatewayResult<Option<Arc<Semaphore>>> {
        let Some(limits) = alias.limits.per_caller else {
            return Ok(None);
        };
        let caller_id = caller_id(caller)?;
        let mut policy = self.policy_lock()?;
        let state = policy.aliases.get_mut(&alias.name).ok_or_else(|| {
            GatewayError::new(
                GatewayErrorKind::Internal,
                format!("missing policy state for alias: {}", alias.name),
            )
        })?;
        if !state.callers.contains_key(caller_id)
            && state.callers.len() >= alias.limits.max_tracked_callers
        {
            return Err(GatewayError::new(
                GatewayErrorKind::Overloaded,
                "model alias caller limit state capacity exceeded",
            ));
        }
        let state = state
            .callers
            .entry(caller_id.to_string())
            .or_insert_with(|| CallerPolicyState::new(limits));
        Ok(state.concurrency.clone())
    }

    fn policy_lock(&self) -> GatewayResult<std::sync::MutexGuard<'_, GatewayPolicyState>> {
        self.policy.lock().map_err(|_| {
            GatewayError::new(
                GatewayErrorKind::Internal,
                "gateway policy state lock is poisoned",
            )
        })
    }
}

impl Default for GatewayAdapter {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for a framework-agnostic gateway adapter.
#[derive(Default)]
pub struct GatewayBuilder {
    aliases: Vec<GatewayAlias>,
}

impl GatewayBuilder {
    /// Add an alias to the adapter being built.
    pub fn alias(mut self, alias: GatewayAlias) -> Self {
        self.aliases.push(alias);
        self
    }

    /// Build the adapter.
    ///
    /// # Errors
    ///
    /// Returns an error when aliases contain duplicates.
    pub fn build(self) -> GatewayResult<GatewayAdapter> {
        let mut adapter = GatewayAdapter::new();
        for alias in self.aliases {
            adapter.add_alias(alias)?;
        }
        Ok(adapter)
    }
}

/// Redacted snapshot of gateway adapter state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewaySnapshot {
    /// Alias snapshots in deterministic name order.
    pub aliases: Vec<GatewayAliasSnapshot>,
}

/// Redacted snapshot of one configured alias.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayAliasSnapshot {
    /// Public alias name.
    pub name: String,
    /// Whether query is enabled.
    pub query: bool,
    /// Whether chat is enabled.
    pub chat: bool,
    /// Whether embedding is enabled.
    pub embed: bool,
    /// Configured alias limits.
    pub limits: GatewayAliasLimits,
    /// Total requests accepted by the global alias scope.
    pub global_total_requests: u64,
    /// Requests accepted in the current global rolling minute.
    pub global_window_requests: u32,
    /// Number of caller IDs currently tracked for per-caller limits.
    pub tracked_callers: usize,
}

#[derive(Default)]
struct GatewayPolicyState {
    aliases: BTreeMap<String, AliasPolicyState>,
}

struct AliasPolicyState {
    global: BucketState,
    callers: BTreeMap<String, CallerPolicyState>,
}

struct CallerPolicyState {
    bucket: BucketState,
    concurrency: Option<Arc<Semaphore>>,
}

impl CallerPolicyState {
    fn new(limits: GatewayRequestLimits) -> Self {
        Self {
            bucket: BucketState {
                window_started: Instant::now(),
                ..BucketState::default()
            },
            concurrency: limits
                .max_concurrent_requests
                .map(|limit| Arc::new(Semaphore::new(limit))),
        }
    }
}

struct BucketState {
    window_started: Instant,
    window_count: u32,
    total_count: u64,
}

impl Default for BucketState {
    fn default() -> Self {
        Self {
            window_started: Instant::now(),
            window_count: 0,
            total_count: 0,
        }
    }
}

impl BucketState {
    fn check_and_increment(
        &mut self,
        limits: GatewayRequestLimits,
        label: &'static str,
    ) -> GatewayResult<()> {
        let now = Instant::now();
        if now.duration_since(self.window_started) >= Duration::from_secs(60) {
            self.window_started = now;
            self.window_count = 0;
        }

        if limits
            .max_requests_total
            .is_some_and(|limit| self.total_count >= limit)
        {
            return Err(GatewayError::new(
                GatewayErrorKind::QuotaExceeded,
                format!("{label} quota exhausted"),
            ));
        }

        if limits
            .max_requests_per_minute
            .is_some_and(|limit| self.window_count >= limit)
        {
            let retry_after = Duration::from_secs(60)
                .saturating_sub(now.duration_since(self.window_started))
                .max(Duration::from_secs(1));
            return Err(GatewayError::new(
                GatewayErrorKind::RateLimited,
                format!("{label} rate limit exceeded"),
            )
            .with_retry_after(Some(retry_after)));
        }

        self.total_count += 1;
        self.window_count += 1;
        Ok(())
    }
}

struct GatewayRequestPermit {
    _global_concurrency: Option<OwnedSemaphorePermit>,
    _caller_concurrency: Option<OwnedSemaphorePermit>,
}

fn hold_permit(
    stream: GatewayStream<GatewayStreamEvent>,
    permit: GatewayRequestPermit,
) -> GatewayStream<GatewayStreamEvent> {
    Box::pin(stream.map(move |event| {
        let _permit = &permit;
        event
    }))
}

fn caller_id(caller: &GatewayCaller) -> GatewayResult<&str> {
    caller.id.as_deref().ok_or_else(|| {
        GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "caller ID is required for per-caller gateway limits",
        )
    })
}

fn validate_gateway_alias_name(alias: &str) -> GatewayResult<()> {
    if alias.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "alias name must not be empty",
        ));
    }
    if alias.trim() != alias {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "alias name must not contain surrounding whitespace",
        ));
    }
    Ok(())
}

fn validate_gateway_access_alias(alias: &str) -> GatewayResult<()> {
    if alias.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "caller access alias name must not be empty",
        ));
    }
    if alias.trim() != alias {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "caller access alias name must not contain surrounding whitespace",
        ));
    }
    Ok(())
}

fn validate_gateway_caller_id(id: &str) -> GatewayResult<()> {
    if id.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "caller ID must not be empty",
        ));
    }
    if id.trim() != id {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "caller ID must not contain surrounding whitespace",
        ));
    }
    Ok(())
}

/// Validate a gateway bearer secret used by host-server authentication.
///
/// # Errors
///
/// Returns an error when the secret is blank or contains whitespace.
pub fn validate_gateway_bearer_secret(secret: &str, field: &str) -> GatewayResult<()> {
    if secret.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not be empty"),
        ));
    }
    if secret.chars().any(char::is_whitespace) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not contain whitespace"),
        ));
    }
    Ok(())
}

/// Compare two secret byte slices without early exit.
pub fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut diff = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let left = left.get(index).copied().unwrap_or_default();
        let right = right.get(index).copied().unwrap_or_default();
        diff |= usize::from(left ^ right);
    }
    diff == 0
}

fn validate_alias_limits(limits: GatewayAliasLimits) -> GatewayResult<()> {
    validate_request_limits(limits.global)?;
    if let Some(limits) = limits.per_caller {
        validate_request_limits(limits)?;
    }
    if limits.max_tracked_callers == 0 {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "max_tracked_callers must be greater than zero",
        ));
    }
    Ok(())
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

impl fmt::Debug for GatewayAlias {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GatewayAlias")
            .field("name", &self.name)
            .field("operations", &self.operations)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "tests/server_tests.rs"]
mod server_tests;
