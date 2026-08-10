//! Unified local, provider, and gateway inference facade.
//!
//! This module owns endpoint resolution and shared request/result envelopes.
//! Local runtime work stays in the engine modules; provider and gateway
//! endpoints are registered through the same client API.

#[allow(clippy::module_inception)]
mod client;
mod dispatch;
pub mod endpoint;
mod error;
mod gateway;
#[cfg(not(target_family = "wasm"))]
mod gateway_endpoint;
#[cfg(not(target_family = "wasm"))]
mod io_executor;
mod local_endpoint;
mod map;
#[cfg(feature = "providers")]
mod provider;
#[cfg(all(feature = "providers", not(target_family = "wasm")))]
mod provider_endpoint;
mod request;
mod response;
mod run;
mod validate;

pub use client::SippClient;
pub use endpoint::{Endpoint, EndpointCapabilities, EndpointRef};
pub use error::{
    EndpointError, ProviderEndpointError, ProviderEndpointErrorKind, SippError, SippResult,
};
pub use gateway::{
    GatewayAuthentication, GatewayDescriptor, GatewayRoutes, GatewaySecret, GatewayTimeoutPolicy,
};
#[cfg(feature = "providers")]
pub use provider::{
    AnthropicProviderConfig, OpenAiCompatibleProviderConfig, OpenAiProviderConfig,
    ProviderAuthConfig, ProviderDescriptor, ProviderSecret,
};
pub use request::{
    LocalEmbedOptions, LocalTextOptions, RequestExtra, SippChatRequest, SippEmbedRequest,
    SippListenRequest, SippQueryRequest, SippRequestContext, SippSpeakRequest, SippTextOptions,
    DEFAULT_TRANSCRIPTION_MAX_TOKENS,
};
pub use response::{
    SippAudioResponse, SippEmbeddingResponse, SippResponseMetadata, SippTextResponse,
};
pub use run::{
    SippAudioResponseFuture, SippAudioRun, SippCancellationHandle, SippCancellationReason,
    SippEmbeddingResponseFuture, SippEmbeddingRun, SippTextResponseFuture, SippTextRun,
    SippTokenBatches,
};

/// Default native registry and managed-asset root for a client.
pub const DEFAULT_STORAGE_ROOT: &str = ".sipp-models";
