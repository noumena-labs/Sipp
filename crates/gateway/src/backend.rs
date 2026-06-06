use std::{collections::BTreeMap, pin::Pin, sync::Arc};

use async_trait::async_trait;
use cogentlm_core::{ChatMessage, FinishReason, TokenBatch, TokenUsage};
use cogentlm_engine::engine::{
    ChatRequest as EngineChatRequest, CogentEngine, EmbedOptions, EmbedRequest,
    EmbeddingResult as EngineEmbeddingResult, GenerationResult as EngineGenerationResult,
    NativeRuntimeConfig, QueryOptions, QueryRequest, RequestSampling, RequestStats,
    SamplingRuntimePatch, DEFAULT_CONTEXT_KEY, DEFAULT_MAX_TOKENS,
};
use cogentlm_providers::{
    ProviderBackend, ProviderChatRequest, ProviderEmbedRequest, ProviderEmbeddingResponse,
    ProviderError, ProviderErrorKind, ProviderGenerateRequest, ProviderGenerationOptions,
    ProviderOptions, ProviderResponse, ProviderStreamEvent, ProviderTextOutput, ProviderTransport,
};
use futures_util::{stream, Stream, StreamExt};
use serde_json::Value;

use crate::{GatewayError, GatewayErrorKind, GatewayResult};

/// Stream returned by gateway text operations.
pub type GatewayStream<T> = Pin<Box<dyn Stream<Item = GatewayResult<T>> + Send>>;

/// Gateway streaming event emitted by query and chat backends.
#[derive(Debug, Clone, PartialEq)]
pub enum GatewayStreamEvent {
    /// Text token batch.
    TokenBatch(TokenBatch),
    /// Token usage.
    Usage { usage: TokenUsage },
    /// Final finish reason.
    Finished { finish_reason: FinishReason },
}

/// Text generation options shared by query and chat.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BackendGenerationOptions {
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff.
    pub top_p: Option<f32>,
    /// Stop strings.
    pub stop: Vec<String>,
}

/// Gateway query request passed to a backend.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendQueryRequest {
    /// Raw prompt.
    pub prompt: String,
    /// Generation options.
    pub options: BackendGenerationOptions,
    /// Gateway-specific options.
    pub gateway_options: BTreeMap<String, Value>,
}

/// Gateway chat request passed to a backend.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendChatRequest {
    /// Chat messages.
    pub messages: Vec<ChatMessage>,
    /// Generation options.
    pub options: BackendGenerationOptions,
    /// Gateway-specific options.
    pub gateway_options: BTreeMap<String, Value>,
}

/// Gateway embedding request passed to a backend.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendEmbedRequest {
    /// Input text.
    pub input: String,
    /// Gateway-specific options.
    pub gateway_options: BTreeMap<String, Value>,
}

/// Normalized text output returned by a gateway backend.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendTextOutput {
    /// Generated text.
    pub text: String,
    /// Normalized finish reason.
    pub finish_reason: FinishReason,
    /// Token usage when available.
    pub usage: Option<TokenUsage>,
    /// Backend response id when available.
    pub response_id: Option<String>,
}

/// Normalized embedding output returned by a gateway backend.
#[derive(Debug, Clone, PartialEq)]
pub struct BackendEmbeddingOutput {
    /// Embedding vector.
    pub values: Vec<f32>,
    /// Token usage when available.
    pub usage: Option<TokenUsage>,
    /// Backend response id when available.
    pub response_id: Option<String>,
}

/// Server-owned local CogentEngine request defaults for a gateway alias.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalCogentEngineOptions {
    /// Context key used for query and chat requests.
    pub context_key: String,
    /// Grammar applied to query and chat requests.
    pub grammar: String,
    /// JSON schema applied to query and chat requests.
    pub json_schema: String,
    /// Context key used for embedding requests.
    pub embedding_context_key: Option<String>,
    /// Whether embedding vectors should be L2-normalized.
    pub normalize_embeddings: bool,
}

impl Default for LocalCogentEngineOptions {
    fn default() -> Self {
        Self {
            context_key: DEFAULT_CONTEXT_KEY.to_string(),
            grammar: String::new(),
            json_schema: String::new(),
            embedding_context_key: None,
            normalize_embeddings: true,
        }
    }
}

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

    /// Enable only selected operations.
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

/// Server-side backend used by a gateway alias.
#[async_trait]
pub trait GatewayBackend: Send + Sync {
    /// Run raw prompt generation.
    async fn query(&self, req: BackendQueryRequest) -> GatewayResult<BackendTextOutput>;

    /// Stream raw prompt generation.
    async fn stream_query(
        &self,
        req: BackendQueryRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>>;

    /// Run chat generation.
    async fn chat(&self, req: BackendChatRequest) -> GatewayResult<BackendTextOutput>;

    /// Stream chat generation.
    async fn stream_chat(
        &self,
        req: BackendChatRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>>;

    /// Run embedding.
    async fn embed(&self, req: BackendEmbedRequest) -> GatewayResult<BackendEmbeddingOutput>;
}

/// Gateway backend backed by an in-process CogentEngine.
#[derive(Clone)]
pub struct LocalCogentEngineBackend {
    engine: CogentEngine,
    options: LocalCogentEngineOptions,
}

impl LocalCogentEngineBackend {
    /// Load a local model and expose it as a gateway backend.
    pub async fn load(
        model_path: impl AsRef<std::path::Path>,
        runtime: NativeRuntimeConfig,
        options: LocalCogentEngineOptions,
    ) -> GatewayResult<Self> {
        validate_local_options(&options)?;
        let engine = CogentEngine::load(model_path, runtime)
            .await
            .map_err(engine_error)?;
        Ok(Self { engine, options })
    }
}

#[async_trait]
impl GatewayBackend for LocalCogentEngineBackend {
    async fn query(&self, req: BackendQueryRequest) -> GatewayResult<BackendTextOutput> {
        reject_local_gateway_options(&req.gateway_options)?;
        let run = self
            .engine
            .query(local_query_request(req, &self.options, false)?);
        run.await.map(local_text_output).map_err(engine_error)
    }

    async fn stream_query(
        &self,
        req: BackendQueryRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        reject_local_gateway_options(&req.gateway_options)?;
        let run = self
            .engine
            .query(local_query_request(req, &self.options, true)?);
        Ok(local_text_stream(run))
    }

    async fn chat(&self, req: BackendChatRequest) -> GatewayResult<BackendTextOutput> {
        reject_local_gateway_options(&req.gateway_options)?;
        let options = local_query_options(req.options, &self.options)?;
        let run = self
            .engine
            .chat(EngineChatRequest::new(req.messages).options(options));
        run.await.map(local_text_output).map_err(engine_error)
    }

    async fn stream_chat(
        &self,
        req: BackendChatRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        reject_local_gateway_options(&req.gateway_options)?;
        let options = local_query_options(req.options, &self.options)?;
        let run = self.engine.chat(
            EngineChatRequest::new(req.messages)
                .options(options)
                .emit_tokens(true),
        );
        Ok(local_text_stream(run))
    }

    async fn embed(&self, req: BackendEmbedRequest) -> GatewayResult<BackendEmbeddingOutput> {
        reject_local_gateway_options(&req.gateway_options)?;
        let run = self
            .engine
            .embed(EmbedRequest {
                input: req.input,
                options: EmbedOptions {
                    normalize: self.options.normalize_embeddings,
                    context_key: self.options.embedding_context_key.clone(),
                },
            })
            .into_response();
        run.await.map(local_embedding_output).map_err(engine_error)
    }
}

/// Deterministic backend for local gateway development and tests.
#[derive(Debug, Clone)]
pub struct MockBackend {
    text: String,
    embedding_dimensions: usize,
}

impl MockBackend {
    /// Create a mock backend with deterministic text and embedding output.
    pub fn new(text: impl Into<String>, embedding_dimensions: usize) -> Self {
        Self {
            text: text.into(),
            embedding_dimensions,
        }
    }
}

#[async_trait]
impl GatewayBackend for MockBackend {
    async fn query(&self, req: BackendQueryRequest) -> GatewayResult<BackendTextOutput> {
        reject_mock_gateway_options(&req.gateway_options)?;
        Ok(BackendTextOutput {
            text: format!("{}{}", self.text, req.prompt),
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage {
                input_tokens: Some(req.prompt.split_whitespace().count() as u32),
                output_tokens: Some(1),
                total_tokens: None,
            }),
            response_id: Some("mock-query".to_string()),
        })
    }

    async fn stream_query(
        &self,
        req: BackendQueryRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        let output = self.query(req).await?;
        Ok(text_stream(output))
    }

    async fn chat(&self, req: BackendChatRequest) -> GatewayResult<BackendTextOutput> {
        reject_mock_gateway_options(&req.gateway_options)?;
        let last = req
            .messages
            .last()
            .map(|message| message.content.as_str())
            .unwrap_or_default();
        Ok(BackendTextOutput {
            text: format!("{}{}", self.text, last),
            finish_reason: FinishReason::Stop,
            usage: Some(TokenUsage {
                input_tokens: Some(req.messages.len() as u32),
                output_tokens: Some(1),
                total_tokens: None,
            }),
            response_id: Some("mock-chat".to_string()),
        })
    }

    async fn stream_chat(
        &self,
        req: BackendChatRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        let output = self.chat(req).await?;
        Ok(text_stream(output))
    }

    async fn embed(&self, req: BackendEmbedRequest) -> GatewayResult<BackendEmbeddingOutput> {
        reject_mock_gateway_options(&req.gateway_options)?;
        let mut values = Vec::with_capacity(self.embedding_dimensions);
        let seed = req
            .input
            .bytes()
            .fold(0_u32, |acc, byte| acc.wrapping_add(u32::from(byte)));
        for index in 0..self.embedding_dimensions {
            values.push(((seed + index as u32) % 997) as f32 / 997.0);
        }
        Ok(BackendEmbeddingOutput {
            values,
            usage: Some(TokenUsage {
                input_tokens: Some(req.input.split_whitespace().count() as u32),
                output_tokens: None,
                total_tokens: None,
            }),
            response_id: Some("mock-embed".to_string()),
        })
    }
}

/// Gateway backend backed by the existing provider transport package.
#[derive(Clone)]
pub struct ProviderGatewayBackend {
    model: String,
    transport: ProviderTransport,
}

impl ProviderGatewayBackend {
    /// Create a backend that maps a public alias to a private provider model.
    ///
    /// Returns an error when the private provider model is blank or has
    /// surrounding whitespace.
    pub fn new(model: impl Into<String>, transport: ProviderTransport) -> GatewayResult<Self> {
        let model = model.into();
        validate_provider_model(&model)?;
        Ok(Self { model, transport })
    }

    /// Create a backend from a provider backend implementation.
    ///
    /// Returns an error when the private provider model is blank or has
    /// surrounding whitespace.
    pub fn from_provider_backend(
        model: impl Into<String>,
        backend: Arc<dyn ProviderBackend>,
    ) -> GatewayResult<Self> {
        Self::new(model, ProviderTransport::from_backend(backend))
    }
}

#[async_trait]
impl GatewayBackend for ProviderGatewayBackend {
    async fn query(&self, req: BackendQueryRequest) -> GatewayResult<BackendTextOutput> {
        reject_provider_gateway_options(&req.gateway_options)?;
        self.transport
            .generate(ProviderGenerateRequest {
                model: self.model.clone(),
                prompt: req.prompt,
                options: provider_generation_options(req.options),
                provider_options: ProviderOptions::default(),
            })
            .await
            .map(provider_text_response)
            .map_err(provider_error)
    }

    async fn stream_query(
        &self,
        req: BackendQueryRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        reject_provider_gateway_options(&req.gateway_options)?;
        let stream = self
            .transport
            .stream_generate(ProviderGenerateRequest {
                model: self.model.clone(),
                prompt: req.prompt,
                options: provider_generation_options(req.options),
                provider_options: ProviderOptions::default(),
            })
            .await
            .map_err(provider_error)?;
        Ok(Box::pin(stream.map(|event| {
            event.map(provider_stream_event).map_err(provider_error)
        })))
    }

    async fn chat(&self, req: BackendChatRequest) -> GatewayResult<BackendTextOutput> {
        reject_provider_gateway_options(&req.gateway_options)?;
        self.transport
            .chat(ProviderChatRequest {
                model: self.model.clone(),
                messages: req.messages,
                options: provider_generation_options(req.options),
                provider_options: ProviderOptions::default(),
            })
            .await
            .map(provider_text_response)
            .map_err(provider_error)
    }

    async fn stream_chat(
        &self,
        req: BackendChatRequest,
    ) -> GatewayResult<GatewayStream<GatewayStreamEvent>> {
        reject_provider_gateway_options(&req.gateway_options)?;
        let stream = self
            .transport
            .stream_chat(ProviderChatRequest {
                model: self.model.clone(),
                messages: req.messages,
                options: provider_generation_options(req.options),
                provider_options: ProviderOptions::default(),
            })
            .await
            .map_err(provider_error)?;
        Ok(Box::pin(stream.map(|event| {
            event.map(provider_stream_event).map_err(provider_error)
        })))
    }

    async fn embed(&self, req: BackendEmbedRequest) -> GatewayResult<BackendEmbeddingOutput> {
        reject_provider_gateway_options(&req.gateway_options)?;
        self.transport
            .embed(ProviderEmbedRequest {
                model: self.model.clone(),
                input: req.input,
                provider_options: ProviderOptions::default(),
            })
            .await
            .map(provider_embedding_response)
            .map_err(provider_error)
    }
}

fn text_stream(output: BackendTextOutput) -> GatewayStream<GatewayStreamEvent> {
    let text = output.text.clone();
    let finish_reason = output.finish_reason;
    let mut events = vec![Ok(GatewayStreamEvent::TokenBatch(TokenBatch {
        request_id: output.response_id.unwrap_or_default(),
        stream_id: 0,
        sequence_start: 0,
        frame_count: 1,
        byte_count: text.len() as u32,
        stats: cogentlm_core::TokenEmissionStats {
            frames_sent: 1,
            bytes_sent: text.len() as u64,
            batches_sent: 1,
        },
        text,
    }))];
    if let Some(usage) = output.usage {
        events.push(Ok(GatewayStreamEvent::Usage { usage }));
    }
    events.push(Ok(GatewayStreamEvent::Finished { finish_reason }));
    Box::pin(stream::iter(events))
}

fn local_query_request(
    request: BackendQueryRequest,
    options: &LocalCogentEngineOptions,
    emit_tokens: bool,
) -> GatewayResult<QueryRequest> {
    Ok(QueryRequest::new(request.prompt)
        .options(local_query_options(request.options, options)?)
        .emit_tokens(emit_tokens))
}

fn local_query_options(
    options: BackendGenerationOptions,
    local: &LocalCogentEngineOptions,
) -> GatewayResult<QueryOptions> {
    let max_tokens = match options.max_tokens {
        Some(max_tokens) => i32::try_from(max_tokens).map_err(|_| {
            GatewayError::new(
                GatewayErrorKind::InvalidRequest,
                "local CogentEngine max_tokens exceeds i32::MAX",
            )
        })?,
        None => DEFAULT_MAX_TOKENS,
    };
    Ok(QueryOptions {
        context_key: local.context_key.clone(),
        max_tokens,
        grammar: local.grammar.clone(),
        json_schema: local.json_schema.clone(),
        stop: options.stop,
        sampling: local_sampling(options.temperature, options.top_p),
        media: Vec::new(),
    })
}

fn local_sampling(temperature: Option<f32>, top_p: Option<f32>) -> Option<RequestSampling> {
    if temperature.is_some() || top_p.is_some() {
        Some(RequestSampling::Patch(SamplingRuntimePatch {
            temperature,
            top_p,
        }))
    } else {
        None
    }
}

fn local_text_stream(
    run: cogentlm_engine::engine::EngineTextRun,
) -> GatewayStream<GatewayStreamEvent> {
    let (tokens, response) = run.into_parts();
    let token_events: GatewayStream<GatewayStreamEvent> = match tokens {
        Some(tokens) => Box::pin(tokens.map(|batch| Ok(GatewayStreamEvent::TokenBatch(batch)))),
        None => Box::pin(stream::empty()),
    };
    let final_events = stream::once(async move {
        response
            .await
            .map(local_final_stream_events)
            .map_err(engine_error)
    })
    .flat_map(|result| match result {
        Ok(events) => stream::iter(events.into_iter().map(Ok).collect::<Vec<_>>()),
        Err(error) => stream::iter(vec![Err(error)]),
    });
    Box::pin(token_events.chain(final_events))
}

fn local_final_stream_events(result: EngineGenerationResult) -> Vec<GatewayStreamEvent> {
    vec![
        GatewayStreamEvent::Usage {
            usage: local_usage_from_stats(result.stats),
        },
        GatewayStreamEvent::Finished {
            finish_reason: result.finish_reason,
        },
    ]
}

fn local_text_output(result: EngineGenerationResult) -> BackendTextOutput {
    BackendTextOutput {
        text: result.text,
        finish_reason: result.finish_reason,
        usage: Some(local_usage_from_stats(result.stats)),
        response_id: Some(result.id),
    }
}

fn local_embedding_output(result: EngineEmbeddingResult) -> BackendEmbeddingOutput {
    BackendEmbeddingOutput {
        values: result.values,
        usage: Some(local_usage_from_stats(result.stats)),
        response_id: Some(result.id),
    }
}

fn local_usage_from_stats(stats: RequestStats) -> TokenUsage {
    let input_tokens = nonnegative_i32_to_u32(stats.input_tokens);
    let output_tokens = nonnegative_i32_to_u32(stats.output_tokens);
    let total_tokens = match (input_tokens, output_tokens) {
        (Some(input), Some(output)) => input.checked_add(output),
        _ => None,
    };
    TokenUsage {
        input_tokens,
        output_tokens,
        total_tokens,
    }
}

fn nonnegative_i32_to_u32(value: i32) -> Option<u32> {
    u32::try_from(value).ok()
}

fn reject_local_gateway_options(options: &BTreeMap<String, Value>) -> GatewayResult<()> {
    if options.is_empty() {
        return Ok(());
    }
    Err(GatewayError::new(
        GatewayErrorKind::InvalidRequest,
        "local CogentEngine backend does not accept gateway_options",
    ))
}

fn reject_provider_gateway_options(options: &BTreeMap<String, Value>) -> GatewayResult<()> {
    if options.is_empty() {
        return Ok(());
    }
    Err(GatewayError::new(
        GatewayErrorKind::InvalidRequest,
        "provider gateway backend does not accept request gateway_options",
    ))
}

fn reject_mock_gateway_options(options: &BTreeMap<String, Value>) -> GatewayResult<()> {
    if options.is_empty() {
        return Ok(());
    }
    Err(GatewayError::new(
        GatewayErrorKind::InvalidRequest,
        "mock gateway backend does not accept gateway_options",
    ))
}

fn validate_provider_model(model: &str) -> GatewayResult<()> {
    if model.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider backend model must not be empty",
        ));
    }
    if model.trim() != model {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "provider backend model must not contain surrounding whitespace",
        ));
    }
    Ok(())
}

fn validate_local_options(options: &LocalCogentEngineOptions) -> GatewayResult<()> {
    if options.context_key.trim().is_empty() {
        return Err(GatewayError::new(
            GatewayErrorKind::InvalidRequest,
            "local CogentEngine context_key must not be empty",
        ));
    }
    if options
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

fn provider_generation_options(options: BackendGenerationOptions) -> ProviderGenerationOptions {
    ProviderGenerationOptions {
        max_tokens: options.max_tokens,
        temperature: options.temperature,
        top_p: options.top_p,
        stop: options.stop,
    }
}

fn provider_text_response(response: ProviderResponse<ProviderTextOutput>) -> BackendTextOutput {
    BackendTextOutput {
        text: response.result.text,
        finish_reason: response.result.finish_reason,
        usage: response.usage,
        response_id: response.metadata.response_id,
    }
}

fn provider_embedding_response(response: ProviderEmbeddingResponse) -> BackendEmbeddingOutput {
    BackendEmbeddingOutput {
        values: response.result.values,
        usage: response.usage,
        response_id: response.metadata.response_id,
    }
}

fn provider_stream_event(event: ProviderStreamEvent) -> GatewayStreamEvent {
    match event {
        ProviderStreamEvent::TokenBatch(batch) => GatewayStreamEvent::TokenBatch(batch),
        ProviderStreamEvent::Usage { usage } => GatewayStreamEvent::Usage { usage },
        ProviderStreamEvent::Finished { finish_reason } => {
            GatewayStreamEvent::Finished { finish_reason }
        }
    }
}

fn provider_error(error: ProviderError) -> GatewayError {
    GatewayError::new(
        provider_error_kind(error.kind),
        provider_error_message(error.kind),
    )
    .with_retry_after(error.retry_after)
}

fn provider_error_message(kind: ProviderErrorKind) -> &'static str {
    match kind {
        ProviderErrorKind::Authentication => "provider authentication failed",
        ProviderErrorKind::Authorization => "provider authorization failed",
        ProviderErrorKind::RateLimited => "provider rate limit exceeded",
        ProviderErrorKind::QuotaExceeded => "provider quota exceeded",
        ProviderErrorKind::InvalidRequest => "provider request is invalid",
        ProviderErrorKind::UnsupportedFeature => "provider backend does not support this operation",
        ProviderErrorKind::ModelNotFound => "provider model was not found",
        ProviderErrorKind::Timeout => "provider request timed out",
        ProviderErrorKind::Overloaded => "provider backend is overloaded",
        ProviderErrorKind::Transport => "provider transport failed",
        ProviderErrorKind::Provider => "provider response was invalid",
    }
}

fn provider_error_kind(kind: ProviderErrorKind) -> GatewayErrorKind {
    match kind {
        ProviderErrorKind::Authentication => GatewayErrorKind::Authentication,
        ProviderErrorKind::Authorization => GatewayErrorKind::Authorization,
        ProviderErrorKind::RateLimited => GatewayErrorKind::RateLimited,
        ProviderErrorKind::QuotaExceeded => GatewayErrorKind::QuotaExceeded,
        ProviderErrorKind::InvalidRequest => GatewayErrorKind::InvalidRequest,
        ProviderErrorKind::UnsupportedFeature => GatewayErrorKind::UnsupportedFeature,
        ProviderErrorKind::ModelNotFound => GatewayErrorKind::ModelNotFound,
        ProviderErrorKind::Timeout => GatewayErrorKind::Timeout,
        ProviderErrorKind::Overloaded => GatewayErrorKind::Overloaded,
        ProviderErrorKind::Transport => GatewayErrorKind::Transport,
        ProviderErrorKind::Provider => GatewayErrorKind::Internal,
    }
}

fn engine_error(error: cogentlm_engine::Error) -> GatewayError {
    let kind = engine_error_kind(&error);
    let message = engine_error_message(&error);
    GatewayError::new(kind, message)
}

fn engine_error_kind(error: &cogentlm_engine::Error) -> GatewayErrorKind {
    match error {
        cogentlm_engine::Error::InvalidRequest(_)
        | cogentlm_engine::Error::InvalidConfig(_)
        | cogentlm_engine::Error::PromptTooLong { .. }
        | cogentlm_engine::Error::BatchCapacity { .. } => GatewayErrorKind::InvalidRequest,
        cogentlm_engine::Error::UnsupportedOperation { .. } => GatewayErrorKind::UnsupportedFeature,
        cogentlm_engine::Error::RuntimeNotReady => GatewayErrorKind::Overloaded,
        cogentlm_engine::Error::ModelLoad { .. } => GatewayErrorKind::ModelNotFound,
        cogentlm_engine::Error::RuntimeCommand(_) => GatewayErrorKind::Overloaded,
        cogentlm_engine::Error::InteriorNul(_)
        | cogentlm_engine::Error::ContextInit
        | cogentlm_engine::Error::NullPointer(_)
        | cogentlm_engine::Error::Tokenize
        | cogentlm_engine::Error::TokenToPiece { .. }
        | cogentlm_engine::Error::Decode(_)
        | cogentlm_engine::Error::SamplerInit => GatewayErrorKind::Internal,
    }
}

fn engine_error_message(error: &cogentlm_engine::Error) -> &'static str {
    match error {
        cogentlm_engine::Error::InvalidRequest(_)
        | cogentlm_engine::Error::PromptTooLong { .. }
        | cogentlm_engine::Error::BatchCapacity { .. } => "local CogentEngine request is invalid",
        cogentlm_engine::Error::InvalidConfig(_) => "local CogentEngine configuration is invalid",
        cogentlm_engine::Error::UnsupportedOperation { .. } => {
            "local CogentEngine backend does not support this operation"
        }
        cogentlm_engine::Error::RuntimeNotReady | cogentlm_engine::Error::RuntimeCommand(_) => {
            "local CogentEngine runtime is overloaded"
        }
        cogentlm_engine::Error::ModelLoad { .. } => "local CogentEngine model was not found",
        cogentlm_engine::Error::InteriorNul(_)
        | cogentlm_engine::Error::ContextInit
        | cogentlm_engine::Error::NullPointer(_)
        | cogentlm_engine::Error::Tokenize
        | cogentlm_engine::Error::TokenToPiece { .. }
        | cogentlm_engine::Error::Decode(_)
        | cogentlm_engine::Error::SamplerInit => "local CogentEngine internal failure",
    }
}

#[cfg(test)]
#[path = "tests/backend_tests.rs"]
mod backend_tests;
