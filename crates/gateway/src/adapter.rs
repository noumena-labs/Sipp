use std::{
    collections::{BTreeMap, VecDeque},
    fmt,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use cogentlm_client::{CogentEmbeddingResponse, CogentTextResponse, CogentTextRun, EndpointRef};
use futures_util::{
    future::{select, Either},
    stream, StreamExt,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::{
    ChatRequestBody, EmbedRequestBody, GatewayError, GatewayErrorKind, GatewayExecutionMetadata,
    GatewayExecutor, GatewayRequestContext, GatewayResult, GatewayStream, GatewayStreamEvent,
    GatewayTextOutput, QueryRequestBody,
};

/// Maximum distinct caller IDs tracked per alias by default.
pub const DEFAULT_MAX_TRACKED_CALLERS: usize = 10_000;

/// Public operation exposed by a gateway alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Operation {
    /// Raw prompt text generation.
    Query,
    /// Message-shaped generation.
    Chat,
    /// Vector embedding.
    Embed,
}

impl Operation {
    /// Stable operation name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Chat => "chat",
            Self::Embed => "embed",
        }
    }
}

/// Enabled operation set for an alias.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationSet {
    query: bool,
    chat: bool,
    embed: bool,
}

impl OperationSet {
    /// Enable every public operation.
    pub const fn all() -> Self {
        Self {
            query: true,
            chat: true,
            embed: true,
        }
    }

    /// Enable selected operations.
    pub fn new(operations: impl IntoIterator<Item = Operation>) -> Self {
        let mut set = Self {
            query: false,
            chat: false,
            embed: false,
        };
        for operation in operations {
            match operation {
                Operation::Query => set.query = true,
                Operation::Chat => set.chat = true,
                Operation::Embed => set.embed = true,
            }
        }
        set
    }

    /// Return whether an operation is enabled.
    pub const fn supports(&self, operation: Operation) -> bool {
        match operation {
            Operation::Query => self.query,
            Operation::Chat => self.chat,
            Operation::Embed => self.embed,
        }
    }

    /// Return whether no operations are enabled.
    pub const fn is_empty(&self) -> bool {
        !self.query && !self.chat && !self.embed
    }
}

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

/// Alias-specific replica-local limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GatewayAliasLimits {
    /// Limits shared by every caller for the alias.
    pub global: GatewayRequestLimits,
    /// Limits tracked independently for each identified caller.
    pub per_caller: Option<GatewayRequestLimits>,
    /// Maximum caller IDs retained for this alias.
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

/// Public alias mapped to a private client endpoint.
#[derive(Clone)]
pub struct GatewayAlias {
    name: String,
    endpoint: EndpointRef,
    operations: OperationSet,
    limits: GatewayAliasLimits,
    global_concurrency: Option<Arc<Semaphore>>,
}

impl GatewayAlias {
    /// Create an alias.
    pub fn new(
        name: impl Into<String>,
        endpoint: EndpointRef,
        operations: OperationSet,
        limits: GatewayAliasLimits,
    ) -> GatewayResult<Self> {
        let name = name.into();
        validate_name(&name, "gateway alias")?;
        if operations.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway alias operations must not be empty",
            ));
        }
        validate_limits(limits.global)?;
        if let Some(per_caller) = limits.per_caller {
            validate_limits(per_caller)?;
            if limits.max_tracked_callers == 0 {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    "max_tracked_callers must be greater than zero",
                ));
            }
        }
        let global_concurrency = limits
            .global
            .max_concurrent_requests
            .map(|limit| Arc::new(Semaphore::new(limit)));
        Ok(Self {
            name,
            endpoint,
            operations,
            limits,
            global_concurrency,
        })
    }

    /// Return the public alias name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the private client endpoint.
    pub fn endpoint(&self) -> &EndpointRef {
        &self.endpoint
    }

    /// Return enabled operations.
    pub fn operations(&self) -> &OperationSet {
        &self.operations
    }

    /// Return configured limits.
    pub fn limits(&self) -> GatewayAliasLimits {
        self.limits
    }
}

impl fmt::Debug for GatewayAlias {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GatewayAlias")
            .field("name", &self.name)
            .field("endpoint", &self.endpoint)
            .field("operations", &self.operations)
            .field("limits", &self.limits)
            .finish_non_exhaustive()
    }
}

/// Alias and operation access granted to a caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayAccess {
    aliases: Option<BTreeMap<String, OperationSet>>,
}

impl GatewayAccess {
    /// Allow every alias and operation.
    pub const fn all() -> Self {
        Self { aliases: None }
    }

    /// Restrict access to selected aliases and operations.
    pub fn new(aliases: impl IntoIterator<Item = (String, OperationSet)>) -> GatewayResult<Self> {
        let mut access = BTreeMap::new();
        for (alias, operations) in aliases {
            validate_name(&alias, "access alias")?;
            if operations.is_empty() || access.insert(alias, operations).is_some() {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    "access aliases must be unique and enable at least one operation",
                ));
            }
        }
        if access.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "access aliases must not be empty",
            ));
        }
        Ok(Self {
            aliases: Some(access),
        })
    }

    /// Return whether this access scope permits an operation.
    pub fn allows(&self, alias: &str, operation: Operation) -> bool {
        self.aliases.as_ref().is_none_or(|aliases| {
            aliases
                .get(alias)
                .is_some_and(|operations| operations.supports(operation))
        })
    }
}

/// Authenticated caller context supplied by the host framework.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayCaller {
    /// Stable caller ID used for per-caller limits.
    pub id: Option<String>,
    /// Alias and operation access scope.
    pub access: GatewayAccess,
}

impl GatewayCaller {
    /// Anonymous caller with full access.
    pub const fn anonymous() -> Self {
        Self {
            id: None,
            access: GatewayAccess::all(),
        }
    }

    /// Identified caller with full access.
    pub fn identified(id: impl Into<String>) -> GatewayResult<Self> {
        let id = id.into();
        validate_name(&id, "caller ID")?;
        Ok(Self {
            id: Some(id),
            access: GatewayAccess::all(),
        })
    }
}

/// Framework-neutral gateway adapter.
#[derive(Clone)]
pub struct GatewayAdapter {
    executor: Arc<dyn GatewayExecutor>,
    aliases: Arc<BTreeMap<String, GatewayAlias>>,
    policy: Arc<Mutex<PolicyState>>,
}

impl GatewayAdapter {
    /// Create a builder over an executor.
    pub fn builder(executor: impl GatewayExecutor + 'static) -> GatewayBuilder {
        GatewayBuilder::new(Arc::new(executor))
    }

    /// Create a builder over a shared executor.
    pub fn builder_shared(executor: Arc<dyn GatewayExecutor>) -> GatewayBuilder {
        GatewayBuilder::new(executor)
    }

    /// Return configured aliases.
    pub fn aliases(&self) -> &BTreeMap<String, GatewayAlias> {
        &self.aliases
    }

    /// Execute a finite query.
    pub async fn query(
        &self,
        context: &GatewayRequestContext,
        body: QueryRequestBody,
    ) -> GatewayResult<GatewayTextOutput> {
        let alias = self.alias(context, &body.model, Operation::Query)?;
        validate_query(&body)?;
        let permit = self.acquire(context, alias)?;
        let run = self
            .executor
            .query(context, body.into_client(alias.endpoint.clone()));
        context.cancellation.register(run.cancellation_handle());
        let result = run.await.map(text_output).map_err(GatewayError::from);
        drop(permit);
        result
    }

    /// Execute a streaming query.
    pub fn stream_query(
        &self,
        context: &GatewayRequestContext,
        body: QueryRequestBody,
    ) -> GatewayResult<GatewayStream> {
        let alias = self.alias(context, &body.model, Operation::Query)?;
        validate_query(&body)?;
        let permit = self.acquire(context, alias)?;
        let run = self
            .executor
            .query(context, body.into_client(alias.endpoint.clone()));
        Ok(text_stream(context, run, permit))
    }

    /// Execute a finite chat request.
    pub async fn chat(
        &self,
        context: &GatewayRequestContext,
        body: ChatRequestBody,
    ) -> GatewayResult<GatewayTextOutput> {
        let alias = self.alias(context, &body.model, Operation::Chat)?;
        validate_chat(&body)?;
        let permit = self.acquire(context, alias)?;
        let run = self
            .executor
            .chat(context, body.into_client(alias.endpoint.clone()));
        context.cancellation.register(run.cancellation_handle());
        let result = run.await.map(text_output).map_err(GatewayError::from);
        drop(permit);
        result
    }

    /// Execute a streaming chat request.
    pub fn stream_chat(
        &self,
        context: &GatewayRequestContext,
        body: ChatRequestBody,
    ) -> GatewayResult<GatewayStream> {
        let alias = self.alias(context, &body.model, Operation::Chat)?;
        validate_chat(&body)?;
        let permit = self.acquire(context, alias)?;
        let run = self
            .executor
            .chat(context, body.into_client(alias.endpoint.clone()));
        Ok(text_stream(context, run, permit))
    }

    /// Execute an embedding request.
    pub async fn embed(
        &self,
        context: &GatewayRequestContext,
        body: EmbedRequestBody,
    ) -> GatewayResult<CogentEmbeddingResponse> {
        let alias = self.alias(context, &body.model, Operation::Embed)?;
        validate_non_empty(&body.input, "input")?;
        let permit = self.acquire(context, alias)?;
        let run = self
            .executor
            .embed(context, body.into_client(alias.endpoint.clone()));
        context.cancellation.register(run.cancellation_handle());
        let result = run.await.map_err(GatewayError::from);
        drop(permit);
        result
    }

    /// Return a redacted replica-local limit snapshot.
    pub fn snapshot(&self) -> GatewayResult<GatewaySnapshot> {
        let policy = self.policy.lock().map_err(|_| {
            GatewayError::new(GatewayErrorKind::Internal, "gateway policy lock failed")
        })?;
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
        context: &GatewayRequestContext,
        model: &str,
        operation: Operation,
    ) -> GatewayResult<&GatewayAlias> {
        validate_non_empty(model, "model")?;
        if !context.caller.access.allows(model, operation) {
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
        context: &GatewayRequestContext,
        alias: &GatewayAlias,
    ) -> GatewayResult<RequestPermit> {
        let global = alias
            .global_concurrency
            .as_ref()
            .map(|semaphore| semaphore.clone().try_acquire_owned())
            .transpose()
            .map_err(|_| overloaded())?;

        let mut policy = self.policy.lock().map_err(|_| {
            GatewayError::new(GatewayErrorKind::Internal, "gateway policy lock failed")
        })?;
        let state = policy.aliases.get_mut(alias.name()).ok_or_else(|| {
            GatewayError::new(GatewayErrorKind::Internal, "gateway alias policy missing")
        })?;
        state.global.charge(alias.limits.global)?;

        let caller = match (alias.limits.per_caller, context.caller.id.as_deref()) {
            (Some(limits), Some(caller_id)) => {
                if !state.callers.contains_key(caller_id)
                    && state.callers.len() >= alias.limits.max_tracked_callers
                {
                    return Err(overloaded());
                }
                let caller = state
                    .callers
                    .entry(caller_id.to_string())
                    .or_insert_with(CallerState::default);
                let permit = caller
                    .semaphore(limits)
                    .map(|semaphore| semaphore.try_acquire_owned())
                    .transpose()
                    .map_err(|_| overloaded())?;
                caller.bucket.charge(limits)?;
                permit
            }
            (Some(_), None) => {
                return Err(GatewayError::new(
                    GatewayErrorKind::InvalidRequest,
                    "identified caller is required for per-caller limits",
                ));
            }
            (None, _) => None,
        };
        Ok(RequestPermit {
            _global: global,
            _caller: caller,
        })
    }
}

/// Builder for an immutable gateway adapter.
pub struct GatewayBuilder {
    executor: Arc<dyn GatewayExecutor>,
    aliases: BTreeMap<String, GatewayAlias>,
}

impl GatewayBuilder {
    fn new(executor: Arc<dyn GatewayExecutor>) -> Self {
        Self {
            executor,
            aliases: BTreeMap::new(),
        }
    }

    /// Add an alias.
    pub fn alias(mut self, alias: GatewayAlias) -> GatewayResult<Self> {
        if self.aliases.insert(alias.name.clone(), alias).is_some() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "duplicate gateway alias",
            ));
        }
        Ok(self)
    }

    /// Build the adapter.
    pub fn build(self) -> GatewayResult<GatewayAdapter> {
        if self.aliases.is_empty() {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "gateway requires at least one alias",
            ));
        }
        let policy = PolicyState {
            aliases: self
                .aliases
                .keys()
                .map(|name| (name.clone(), AliasState::default()))
                .collect(),
        };
        Ok(GatewayAdapter {
            executor: self.executor,
            aliases: Arc::new(self.aliases),
            policy: Arc::new(Mutex::new(policy)),
        })
    }
}

/// Redacted adapter snapshot.
#[derive(Debug, Clone)]
pub struct GatewaySnapshot {
    /// Alias snapshots.
    pub aliases: Vec<GatewayAliasSnapshot>,
}

/// Redacted alias snapshot.
#[derive(Debug, Clone)]
pub struct GatewayAliasSnapshot {
    /// Alias name.
    pub name: String,
    /// Whether query is enabled.
    pub query: bool,
    /// Whether chat is enabled.
    pub chat: bool,
    /// Whether embed is enabled.
    pub embed: bool,
    /// Configured replica-local limits.
    pub limits: GatewayAliasLimits,
    /// Total requests charged since startup.
    pub global_total_requests: u64,
    /// Requests charged in the current minute.
    pub global_window_requests: u32,
    /// Caller IDs currently tracked.
    pub tracked_callers: usize,
}

#[derive(Default)]
struct PolicyState {
    aliases: BTreeMap<String, AliasState>,
}

#[derive(Default)]
struct AliasState {
    global: BucketState,
    callers: BTreeMap<String, CallerState>,
}

#[derive(Default)]
struct CallerState {
    bucket: BucketState,
    concurrency: Option<Arc<Semaphore>>,
}

impl CallerState {
    fn semaphore(&mut self, limits: GatewayRequestLimits) -> Option<Arc<Semaphore>> {
        let limit = limits.max_concurrent_requests?;
        Some(
            self.concurrency
                .get_or_insert_with(|| Arc::new(Semaphore::new(limit)))
                .clone(),
        )
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
    fn charge(&mut self, limits: GatewayRequestLimits) -> GatewayResult<()> {
        if self.window_started.elapsed() >= Duration::from_secs(60) {
            self.window_started = Instant::now();
            self.window_count = 0;
        }
        if limits
            .max_requests_per_minute
            .is_some_and(|limit| self.window_count >= limit)
        {
            return Err(GatewayError::new(
                GatewayErrorKind::RateLimited,
                "request rate limit exceeded",
            )
            .with_retry_after(Some(Duration::from_secs(60))));
        }
        if limits
            .max_requests_total
            .is_some_and(|limit| self.total_count >= limit)
        {
            return Err(GatewayError::new(
                GatewayErrorKind::QuotaExceeded,
                "request quota exceeded",
            ));
        }
        self.window_count = self.window_count.saturating_add(1);
        self.total_count = self.total_count.saturating_add(1);
        Ok(())
    }
}

struct RequestPermit {
    _global: Option<OwnedSemaphorePermit>,
    _caller: Option<OwnedSemaphorePermit>,
}

struct TextStreamState {
    tokens: cogentlm_client::CogentTokenBatches,
    response: Option<cogentlm_client::CogentTextResponseFuture>,
    pending: VecDeque<GatewayResult<GatewayStreamEvent>>,
    terminal: bool,
    permit: Option<RequestPermit>,
}

fn text_stream(
    context: &GatewayRequestContext,
    run: CogentTextRun,
    permit: RequestPermit,
) -> GatewayStream {
    let (tokens, response, cancellation) = run.into_parts_with_cancel();
    context.cancellation.register(cancellation);
    let state = TextStreamState {
        tokens,
        response: Some(response),
        pending: VecDeque::new(),
        terminal: false,
        permit: Some(permit),
    };
    Box::pin(stream::unfold(state, |mut state| async move {
        if let Some(event) = state.pending.pop_front() {
            return Some((event, state));
        }
        if state.terminal {
            return None;
        }
        let response = state.response.take()?;
        match select(state.tokens.next(), response).await {
            Either::Left((Some(batch), response)) => {
                state.response = Some(response);
                Some((Ok(GatewayStreamEvent::TokenBatch(batch)), state))
            }
            Either::Left((None, response)) => {
                finish_text_stream(&mut state, response.await);
                state.pending.pop_front().map(|event| (event, state))
            }
            Either::Right((response, token)) => {
                drop(token);
                finish_text_stream(&mut state, response);
                state.pending.pop_front().map(|event| (event, state))
            }
        }
    }))
}

fn finish_text_stream(
    state: &mut TextStreamState,
    response: cogentlm_client::CogentResult<CogentTextResponse>,
) {
    state.terminal = true;
    state.permit.take();
    match response {
        Ok(response) => {
            if let Some(usage) = response.usage {
                state
                    .pending
                    .push_back(Ok(GatewayStreamEvent::Usage { usage }));
            }
            state.pending.push_back(Ok(GatewayStreamEvent::Finished {
                finish_reason: response.finish_reason,
                metadata: response.metadata.into(),
            }));
        }
        Err(error) => {
            state.pending.push_back(Err(error.into()));
        }
    }
}

fn text_output(response: CogentTextResponse) -> GatewayTextOutput {
    GatewayTextOutput {
        text: response.text,
        finish_reason: response.finish_reason,
        usage: response.usage,
        metadata: GatewayExecutionMetadata::from(response.metadata),
    }
}

fn validate_query(body: &QueryRequestBody) -> GatewayResult<()> {
    validate_non_empty(&body.prompt, "prompt")?;
    validate_text_options(
        body.max_tokens,
        body.temperature,
        body.top_p,
        &body.gateway_options,
    )
}

fn validate_chat(body: &ChatRequestBody) -> GatewayResult<()> {
    if body.messages.is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "messages must not be empty",
        ));
    }
    validate_text_options(
        body.max_tokens,
        body.temperature,
        body.top_p,
        &body.gateway_options,
    )
}

fn validate_text_options(
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    gateway_options: &crate::GatewayOptions,
) -> GatewayResult<()> {
    if max_tokens == Some(0) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "max_tokens must be greater than zero",
        ));
    }
    if temperature.is_some_and(|value| !value.is_finite() || value < 0.0) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "temperature must be finite and nonnegative",
        ));
    }
    if top_p.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "top_p must be finite and between 0 and 1",
        ));
    }
    for key in gateway_options.keys() {
        if matches!(
            key.as_str(),
            "context_key"
                | "contextKey"
                | "grammar"
                | "json_schema"
                | "jsonSchema"
                | "sampling"
                | "media"
                | "normalize"
                | "local"
        ) {
            return Err(GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                format!("gateway request cannot contain local-only field: {key}"),
            ));
        }
    }
    Ok(())
}

fn validate_non_empty(value: &str, field: &'static str) -> GatewayResult<()> {
    if value.trim().is_empty() {
        Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must not be empty"),
        ))
    } else {
        Ok(())
    }
}

fn validate_name(value: &str, field: &'static str) -> GatewayResult<()> {
    if value.is_empty() || value.trim() != value || value.len() > 256 {
        Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            format!("{field} must be a non-empty trimmed value"),
        ))
    } else {
        Ok(())
    }
}

fn validate_limits(limits: GatewayRequestLimits) -> GatewayResult<()> {
    if limits.max_concurrent_requests == Some(0)
        || limits.max_requests_per_minute == Some(0)
        || limits.max_requests_total == Some(0)
    {
        Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "gateway limits must be greater than zero",
        ))
    } else {
        Ok(())
    }
}

fn overloaded() -> GatewayError {
    GatewayError::new(
        GatewayErrorKind::Overloaded,
        "gateway concurrency limit exceeded",
    )
}

#[cfg(test)]
#[path = "tests/adapter_tests.rs"]
mod adapter_tests;
