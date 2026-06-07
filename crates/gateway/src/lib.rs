//! Framework-neutral building blocks for CogentLM gateway adapters.
//!
//! This crate owns protocol envelopes, alias routing, access scopes,
//! replica-local limits, cancellation, and execution over `cogentlm-client`.
//! It deliberately does not own listeners, authentication mechanisms, CORS,
//! configuration files, logging, or metrics.

mod adapter;
mod context;
mod error;
mod executor;
mod protocol;

pub use adapter::{
    GatewayAccess, GatewayAdapter, GatewayAlias, GatewayAliasLimits, GatewayAliasSnapshot,
    GatewayBuilder, GatewayCaller, GatewayRequestLimits, GatewaySnapshot, Operation, OperationSet,
    DEFAULT_MAX_TRACKED_CALLERS,
};
pub use context::{
    validate_request_id, GatewayCancellation, GatewayCancellationReason, GatewayRequestContext,
    MAX_REQUEST_ID_BYTES,
};
pub use error::{GatewayError, GatewayErrorKind, GatewayResult};
pub use executor::{CogentClientExecutor, GatewayExecutor};
pub use protocol::{
    finish_reason, ChatMessageBody, ChatRequestBody, EmbedRequestBody, EmbeddingResponseBody,
    ErrorBody, ErrorEnvelope, GatewayExecutionMetadata, GatewayOptions, GatewayStream,
    GatewayStreamEvent, GatewayTextOutput, QueryRequestBody, TextResponseBody, UsageBody,
};
