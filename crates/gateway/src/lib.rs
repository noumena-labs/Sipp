//! Framework-agnostic CogentLM gateway adapter.
//!
//! This crate owns alias routing, request validation, backend execution, and
//! gateway policy. Host applications authenticate callers and then pass a
//! `GatewayCaller` into the adapter. Runnable HTTP server behavior lives in
//! `apps/gateway-server`.

mod backend;
mod config;
mod error;
mod protocol;
mod server;

pub use backend::{
    BackendChatRequest, BackendEmbedRequest, BackendEmbeddingOutput, BackendGenerationOptions,
    BackendQueryRequest, BackendTextOutput, GatewayBackend, GatewayStream, GatewayStreamEvent,
    LocalCogentEngineBackend, LocalCogentEngineOptions, MockBackend, Operation, OperationSet,
    ProviderGatewayBackend,
};
pub use config::{GatewayFileConfig, OperationFileConfig};
pub use error::{GatewayError, GatewayErrorKind, GatewayResult};
pub use protocol::{
    finish_reason, ChatMessageBody, ChatRequestBody, EmbedRequestBody, EmbeddingResponseBody,
    GatewayOptions, QueryRequestBody, TextResponseBody, UsageBody,
};
pub use server::{
    constant_time_eq, validate_gateway_bearer_secret, GatewayAccess, GatewayAdapter, GatewayAlias,
    GatewayAliasLimits, GatewayAliasSnapshot, GatewayBuilder, GatewayCaller, GatewayRequestLimits,
    GatewaySnapshot, DEFAULT_MAX_TRACKED_CALLERS,
};
