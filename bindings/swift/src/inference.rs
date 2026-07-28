//! UniFFI projection for inference requests, runs, and response values.

use std::sync::Arc;

use futures::StreamExt;
use sipp::core::{
    ChatMessage as CoreChatMessage, ChatRole as CoreChatRole, FinishReason as CoreFinishReason,
    TokenBatch as CoreTokenBatch, TokenEmissionStats as CoreTokenEmissionStats,
    TokenUsage as CoreTokenUsage,
};
use sipp::engine::{
    CacheSource as CoreCacheSource, KvReuseMode as CoreCacheMode, PoolingType as CorePoolingType,
    RequestStats as CoreRequestStats,
};
use sipp::{
    EndpointRef as CoreEndpointRef, LocalEmbedOptions as CoreLocalEmbedOptions,
    LocalTextOptions as CoreLocalTextOptions, SippCancellationHandle as CoreCancellationHandle,
    SippCancellationReason as CoreCancellationReason, SippChatRequest as CoreChatRequest,
    SippEmbedRequest as CoreEmbedRequest, SippEmbeddingResponse as CoreEmbeddingResponse,
    SippEmbeddingResponseFuture as CoreEmbeddingResponseFuture,
    SippEmbeddingRun as CoreEmbeddingRun, SippQueryRequest as CoreQueryRequest,
    SippRequestContext as CoreRequestContext, SippResponseMetadata as CoreResponseMetadata,
    SippTextOptions as CoreTextOptions, SippTextResponse as CoreTextResponse,
    SippTextResponseFuture as CoreTextResponseFuture, SippTextRun as CoreTextRun,
    SippTokenBatches as CoreTokenBatches,
};
use tokio::sync::Mutex;

use crate::bridge::FfiError;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/inference_tests.rs"]
mod inference_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Role attached to a chat message across the internal Swift FFI boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Enum)]
pub enum FfiChatRole {
    /// System instruction.
    System,
    /// User input.
    User,
    /// Assistant output supplied as conversation history.
    Assistant,
}

impl From<FfiChatRole> for CoreChatRole {
    fn from(value: FfiChatRole) -> Self {
        match value {
            FfiChatRole::System => Self::System,
            FfiChatRole::User => Self::User,
            FfiChatRole::Assistant => Self::Assistant,
        }
    }
}

/// Role/content message accepted by internal chat requests.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiChatMessage {
    /// Message role.
    pub role: FfiChatRole,
    /// Message text.
    pub content: String,
}

impl From<FfiChatMessage> for CoreChatMessage {
    fn from(value: FfiChatMessage) -> Self {
        Self::new(value.role.into(), value.content)
    }
}

/// Shared generation options accepted by internal text requests.
#[derive(Clone, Debug, PartialEq, uniffi::Record)]
pub struct FfiTextOptions {
    /// Maximum generated token count.
    pub max_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus sampling cutoff.
    pub top_p: Option<f32>,
    /// Stop strings.
    pub stop: Vec<String>,
}

impl From<FfiTextOptions> for CoreTextOptions {
    fn from(value: FfiTextOptions) -> Self {
        Self {
            max_tokens: value.max_tokens,
            temperature: value.temperature,
            top_p: value.top_p,
            stop: value.stop,
        }
    }
}

/// Local-only options accepted by internal text requests.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiLocalTextOptions {
    /// Local KV-cache context key.
    pub context_key: Option<String>,
    /// Grammar constraint.
    pub grammar: Option<String>,
    /// JSON schema constraint.
    pub json_schema: Option<String>,
    /// Binary media payloads for multimodal requests.
    pub media: Vec<Vec<u8>>,
}

impl From<FfiLocalTextOptions> for CoreLocalTextOptions {
    fn from(value: FfiLocalTextOptions) -> Self {
        Self {
            context_key: value.context_key,
            grammar: value.grammar,
            json_schema: value.json_schema,
            sampling: None,
            media: value.media,
        }
    }
}

/// Local-only options accepted by internal embedding requests.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiLocalEmbedOptions {
    /// Local KV-cache context key.
    pub context_key: Option<String>,
    /// Whether to L2-normalize the vector.
    pub normalize: Option<bool>,
}

impl From<FfiLocalEmbedOptions> for CoreLocalEmbedOptions {
    fn from(value: FfiLocalEmbedOptions) -> Self {
        Self {
            context_key: value.context_key,
            normalize: value.normalize,
        }
    }
}

/// Raw-prompt generation request accepted by the internal Swift bridge.
#[derive(Clone, Debug, PartialEq, uniffi::Record)]
pub struct FfiQueryRequest {
    /// Optional application request identifier.
    pub request_id: Option<String>,
    /// Registered endpoint identifier, or the single compatible local endpoint.
    pub endpoint: Option<String>,
    /// Raw prompt text.
    pub prompt: String,
    /// Shared generation options.
    pub options: FfiTextOptions,
    /// Local-only generation options.
    pub local: FfiLocalTextOptions,
    /// Whether the run emits token batches.
    pub emit_tokens: bool,
}

impl FfiQueryRequest {
    pub(crate) fn into_core(self) -> (CoreRequestContext, CoreQueryRequest) {
        let context = CoreRequestContext {
            request_id: self.request_id,
        };
        let request = CoreQueryRequest {
            endpoint: self.endpoint.map(CoreEndpointRef::from_id),
            prompt: self.prompt,
            options: self.options.into(),
            local: self.local.into(),
            extra: Default::default(),
            emit_tokens: self.emit_tokens,
        };
        (context, request)
    }
}

/// Chat generation request accepted by the internal Swift bridge.
#[derive(Clone, Debug, PartialEq, uniffi::Record)]
pub struct FfiChatRequest {
    /// Optional application request identifier.
    pub request_id: Option<String>,
    /// Registered endpoint identifier, or the single compatible local endpoint.
    pub endpoint: Option<String>,
    /// Ordered conversation messages.
    pub messages: Vec<FfiChatMessage>,
    /// Shared generation options.
    pub options: FfiTextOptions,
    /// Local-only generation options.
    pub local: FfiLocalTextOptions,
    /// Whether the run emits token batches.
    pub emit_tokens: bool,
}

impl FfiChatRequest {
    pub(crate) fn into_core(self) -> (CoreRequestContext, CoreChatRequest) {
        let context = CoreRequestContext {
            request_id: self.request_id,
        };
        let request = CoreChatRequest {
            endpoint: self.endpoint.map(CoreEndpointRef::from_id),
            messages: self
                .messages
                .into_iter()
                .map(CoreChatMessage::from)
                .collect(),
            options: self.options.into(),
            local: self.local.into(),
            extra: Default::default(),
            emit_tokens: self.emit_tokens,
        };
        (context, request)
    }
}

/// Embedding request accepted by the internal Swift bridge.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiEmbedRequest {
    /// Optional application request identifier.
    pub request_id: Option<String>,
    /// Registered endpoint identifier, or the single compatible local endpoint.
    pub endpoint: Option<String>,
    /// Input text.
    pub input: String,
    /// Local-only embedding options.
    pub local: FfiLocalEmbedOptions,
}

impl FfiEmbedRequest {
    pub(crate) fn into_core(self) -> (CoreRequestContext, CoreEmbedRequest) {
        let context = CoreRequestContext {
            request_id: self.request_id,
        };
        let request = CoreEmbedRequest {
            endpoint: self.endpoint.map(CoreEndpointRef::from_id),
            input: self.input,
            local: self.local.into(),
            extra: Default::default(),
        };
        (context, request)
    }
}

/// Completion finish reason returned by an internal text response.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Enum)]
pub enum FfiFinishReason {
    /// Endpoint emitted a natural stop.
    Stop,
    /// Requested token limit was reached.
    Length,
    /// Run was cancelled.
    Cancelled,
    /// Endpoint stopped after an inference error.
    Error,
}

impl From<CoreFinishReason> for FfiFinishReason {
    fn from(value: CoreFinishReason) -> Self {
        match value {
            CoreFinishReason::Stop => Self::Stop,
            CoreFinishReason::Length => Self::Length,
            CoreFinishReason::Cancelled => Self::Cancelled,
            CoreFinishReason::Error => Self::Error,
        }
    }
}

/// Token accounting returned by an inference endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiTokenUsage {
    /// Input token count when reported.
    pub input_tokens: Option<u32>,
    /// Output token count when reported.
    pub output_tokens: Option<u32>,
    /// Total token count when reported.
    pub total_tokens: Option<u32>,
}

impl From<CoreTokenUsage> for FfiTokenUsage {
    fn from(value: CoreTokenUsage) -> Self {
        Self {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            total_tokens: value.total_tokens,
        }
    }
}

/// Local KV-cache reuse mode reported by request statistics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Enum)]
pub enum FfiCacheMode {
    /// Prefix reuse is disabled.
    Disabled,
    /// Reuse a live slot prefix.
    LiveSlotPrefix,
    /// Restore a serialized state snapshot.
    StateSnapshot,
    /// Use both live slots and snapshots.
    LiveSlotAndSnapshot,
}

impl From<CoreCacheMode> for FfiCacheMode {
    fn from(value: CoreCacheMode) -> Self {
        match value {
            CoreCacheMode::Disabled => Self::Disabled,
            CoreCacheMode::LiveSlotPrefix => Self::LiveSlotPrefix,
            CoreCacheMode::StateSnapshot => Self::StateSnapshot,
            CoreCacheMode::LiveSlotAndSnapshot => Self::LiveSlotAndSnapshot,
        }
    }
}

/// Local KV-cache source reported by request statistics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Enum)]
pub enum FfiCacheSource {
    /// No cached prefix was used.
    None,
    /// A live slot supplied the prefix.
    Live,
    /// A serialized snapshot supplied the prefix.
    Snapshot,
}

impl From<CoreCacheSource> for FfiCacheSource {
    fn from(value: CoreCacheSource) -> Self {
        match value {
            CoreCacheSource::None => Self::None,
            CoreCacheSource::Live => Self::Live,
            CoreCacheSource::Snapshot => Self::Snapshot,
        }
    }
}

/// Local runtime timing and cache statistics for a completed request.
#[derive(Clone, Copy, Debug, PartialEq, uniffi::Record)]
pub struct FfiRequestStats {
    /// Input token count.
    pub input_tokens: i32,
    /// Output token count.
    pub output_tokens: i32,
    /// Configured cache reuse mode.
    pub cache_mode: FfiCacheMode,
    /// Cache source used by this request.
    pub cache_source: FfiCacheSource,
    /// Reused token count.
    pub cache_hits: i32,
    /// Tokens evaluated during prefill.
    pub prefill_tokens: i32,
    /// Time to first token in milliseconds.
    pub ttft_ms: Option<f64>,
    /// Average inter-token latency in milliseconds.
    pub inter_token_ms: Option<f64>,
    /// End-to-end latency in milliseconds.
    pub e2e_ms: Option<f64>,
    /// End-to-end output-token throughput.
    pub e2e_tokens_per_second: Option<f64>,
    /// Decode-only output-token throughput.
    pub decode_tokens_per_second: Option<f64>,
    /// Prefill throughput.
    pub prefill_tokens_per_second: Option<f64>,
    /// Prefill latency in milliseconds.
    pub prefill_ms: f64,
    /// Decode latency in milliseconds.
    pub decode_ms: f64,
}

impl From<CoreRequestStats> for FfiRequestStats {
    fn from(value: CoreRequestStats) -> Self {
        Self {
            input_tokens: value.input_tokens,
            output_tokens: value.output_tokens,
            cache_mode: value.cache_mode.into(),
            cache_source: value.cache_source.into(),
            cache_hits: value.cache_hits,
            prefill_tokens: value.prefill_tokens,
            ttft_ms: value.ttft_ms,
            inter_token_ms: value.inter_token_ms,
            e2e_ms: value.e2e_ms,
            e2e_tokens_per_second: value.e2e_tokens_per_second,
            decode_tokens_per_second: value.decode_tokens_per_second,
            prefill_tokens_per_second: value.prefill_tokens_per_second,
            prefill_ms: value.prefill_ms,
            decode_ms: value.decode_ms,
        }
    }
}

/// Request and upstream correlation metadata.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiResponseMetadata {
    /// Application request identifier.
    pub request_id: Option<String>,
    /// Upstream request identifier.
    pub upstream_request_id: Option<String>,
    /// Upstream response identifier.
    pub upstream_response_id: Option<String>,
}

impl From<CoreResponseMetadata> for FfiResponseMetadata {
    fn from(value: CoreResponseMetadata) -> Self {
        Self {
            request_id: value.request_id,
            upstream_request_id: value.upstream_request_id,
            upstream_response_id: value.upstream_response_id,
        }
    }
}

/// Final response from an internal query or chat run.
#[derive(Clone, Debug, PartialEq, uniffi::Record)]
pub struct FfiTextResponse {
    /// Endpoint that produced the response.
    pub endpoint: String,
    /// Generated text.
    pub text: String,
    /// Completion finish reason.
    pub finish_reason: FfiFinishReason,
    /// Token usage when reported.
    pub usage: Option<FfiTokenUsage>,
    /// Local runtime statistics for local endpoints.
    pub local_stats: Option<FfiRequestStats>,
    /// Correlation metadata.
    pub metadata: FfiResponseMetadata,
}

impl From<CoreTextResponse> for FfiTextResponse {
    fn from(value: CoreTextResponse) -> Self {
        Self {
            endpoint: value.endpoint.id().to_owned(),
            text: value.text,
            finish_reason: value.finish_reason.into(),
            usage: value.usage.map(FfiTokenUsage::from),
            local_stats: value.local_stats.map(FfiRequestStats::from),
            metadata: value.metadata.into(),
        }
    }
}

/// Embedding pooling strategy reported by a local endpoint.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Enum)]
pub enum FfiPoolingType {
    /// Model default was not resolved.
    Unspecified,
    /// No pooling.
    None,
    /// Mean pooling.
    Mean,
    /// Classification-token pooling.
    Cls,
    /// Last-token pooling.
    Last,
    /// Ranking pooling.
    Rank,
}

impl From<CorePoolingType> for FfiPoolingType {
    fn from(value: CorePoolingType) -> Self {
        match value {
            CorePoolingType::Unspecified => Self::Unspecified,
            CorePoolingType::None => Self::None,
            CorePoolingType::Mean => Self::Mean,
            CorePoolingType::Cls => Self::Cls,
            CorePoolingType::Last => Self::Last,
            CorePoolingType::Rank => Self::Rank,
        }
    }
}

/// Final response from an internal embedding run.
#[derive(Clone, Debug, PartialEq, uniffi::Record)]
pub struct FfiEmbeddingResponse {
    /// Endpoint that produced the response.
    pub endpoint: String,
    /// Embedding vector.
    pub values: Vec<f32>,
    /// Token usage when reported.
    pub usage: Option<FfiTokenUsage>,
    /// Local runtime statistics for local endpoints.
    pub local_stats: Option<FfiRequestStats>,
    /// Pooling used by the endpoint.
    pub pooling: Option<FfiPoolingType>,
    /// Whether the endpoint normalized the vector.
    pub normalized: Option<bool>,
    /// Correlation metadata.
    pub metadata: FfiResponseMetadata,
}

impl From<CoreEmbeddingResponse> for FfiEmbeddingResponse {
    fn from(value: CoreEmbeddingResponse) -> Self {
        Self {
            endpoint: value.endpoint.id().to_owned(),
            values: value.values,
            usage: value.usage.map(FfiTokenUsage::from),
            local_stats: value.local_stats.map(FfiRequestStats::from),
            pooling: value.pooling.map(FfiPoolingType::from),
            normalized: value.normalized,
            metadata: value.metadata.into(),
        }
    }
}

/// Aggregate counters attached to an emitted token batch.
#[derive(Clone, Copy, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiTokenEmissionStats {
    /// Token frames emitted so far.
    pub frames_sent: u64,
    /// UTF-8 payload bytes emitted so far.
    pub bytes_sent: u64,
    /// Token batches emitted so far.
    pub batches_sent: u64,
}

impl From<CoreTokenEmissionStats> for FfiTokenEmissionStats {
    fn from(value: CoreTokenEmissionStats) -> Self {
        Self {
            frames_sent: value.frames_sent,
            bytes_sent: value.bytes_sent,
            batches_sent: value.batches_sent,
        }
    }
}

/// Streaming token payload emitted by an internal text run.
#[derive(Clone, Debug, Eq, PartialEq, uniffi::Record)]
pub struct FfiTokenBatch {
    /// Stable request identifier.
    pub request_id: String,
    /// Numeric local stream identifier.
    pub stream_id: u32,
    /// Sequence index of the first frame.
    pub sequence_start: u32,
    /// Concatenated token text.
    pub text: String,
    /// Token frames represented by this batch.
    pub frame_count: u32,
    /// UTF-8 payload bytes represented by this batch.
    pub byte_count: u32,
    /// Cumulative token-emission counters.
    pub stats: FfiTokenEmissionStats,
}

impl From<CoreTokenBatch> for FfiTokenBatch {
    fn from(value: CoreTokenBatch) -> Self {
        Self {
            request_id: value.request_id,
            stream_id: value.stream_id,
            sequence_start: value.sequence_start,
            text: value.text,
            frame_count: value.frame_count,
            byte_count: value.byte_count,
            stats: value.stats.into(),
        }
    }
}

type SharedTextResponse = Arc<Mutex<Option<CoreTextResponseFuture>>>;
type SharedEmbeddingResponse = Arc<Mutex<Option<CoreEmbeddingResponseFuture>>>;
type SharedTokenBatches = Arc<Mutex<Option<CoreTokenBatches>>>;

/// Internal text run exposed to generated Swift bindings.
#[derive(uniffi::Object)]
pub struct FfiTextRun {
    response: SharedTextResponse,
    tokens: SharedTokenBatches,
    cancellation: CoreCancellationHandle,
}

impl FfiTextRun {
    pub(crate) fn from_core(run: CoreTextRun) -> Self {
        let (tokens, response, cancellation) = run.into_parts_with_cancel();
        Self {
            response: Arc::new(Mutex::new(Some(response))),
            tokens: Arc::new(Mutex::new(Some(tokens))),
            cancellation,
        }
    }
}

#[uniffi::export]
impl FfiTextRun {
    /// Await and consume the run's final response future.
    ///
    /// # Errors
    ///
    /// Returns a typed inference error, or `InvalidState` after prior consumption.
    pub async fn take_response(&self) -> Result<FfiTextResponse, FfiError> {
        let response = self
            .response
            .lock()
            .await
            .take()
            .ok_or_else(|| FfiError::InvalidState {
                message: "text response already consumed".to_owned(),
            })?;
        response
            .await
            .map(FfiTextResponse::from)
            .map_err(FfiError::from)
    }

    /// Await and consume the next token batch, or return `None` at stream end.
    pub async fn next_token(&self) -> Option<FfiTokenBatch> {
        let mut slot = self.tokens.lock().await;
        let Some(tokens) = slot.as_mut() else {
            return None;
        };
        match tokens.next().await {
            Some(batch) => Some(batch.into()),
            None => {
                *slot = None;
                None
            }
        }
    }

    /// Cancel the native run as an explicit caller cancellation.
    pub fn cancel(&self) {
        self.cancellation
            .cancel(CoreCancellationReason::CallerCancelled);
    }
}

/// Internal embedding run exposed to generated Swift bindings.
#[derive(uniffi::Object)]
pub struct FfiEmbeddingRun {
    response: SharedEmbeddingResponse,
    cancellation: CoreCancellationHandle,
}

impl FfiEmbeddingRun {
    pub(crate) fn from_core(run: CoreEmbeddingRun) -> Self {
        let (response, cancellation) = run.into_parts();
        Self {
            response: Arc::new(Mutex::new(Some(response))),
            cancellation,
        }
    }
}

#[uniffi::export]
impl FfiEmbeddingRun {
    /// Await and consume the run's final response future.
    ///
    /// # Errors
    ///
    /// Returns a typed inference error, or `InvalidState` after prior consumption.
    pub async fn take_response(&self) -> Result<FfiEmbeddingResponse, FfiError> {
        let response = self
            .response
            .lock()
            .await
            .take()
            .ok_or_else(|| FfiError::InvalidState {
                message: "embedding response already consumed".to_owned(),
            })?;
        response
            .await
            .map(FfiEmbeddingResponse::from)
            .map_err(FfiError::from)
    }

    /// Cancel the native run as an explicit caller cancellation.
    pub fn cancel(&self) {
        self.cancellation
            .cancel(CoreCancellationReason::CallerCancelled);
    }
}
