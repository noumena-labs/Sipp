//! Server-side CogentLM Remote Gateway Protocol implementation.
//!
//! This crate owns the HTTP boundary for remote inference. It accepts the
//! CogentLM gateway protocol from app-facing clients, authenticates the caller,
//! checks alias capabilities, and routes to server-side backends or provider
//! adapters.

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
pub use config::{GatewayFileConfig, GatewayServerConfig};
pub use error::{GatewayError, GatewayErrorKind, GatewayResult};
pub use server::{
    GatewayAccess, GatewayAlias, GatewayAliasLimits, GatewayService, GatewayServiceLimits,
    GatewayState, GatewayToken,
};
