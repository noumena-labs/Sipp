//! Unified local, provider, and gateway inference facade.
//!
//! The crate owns endpoint resolution and shared request/result envelopes.
//! Local runtime work stays in `cogentlm-engine`; provider and gateway
//! endpoints are registered through the same client API.

mod client;
mod descriptor;
mod dispatch;
mod endpoint;
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

pub use client::CogentClient;
pub use descriptor::{EndpointDescriptor, LocalModelDescriptor};
pub use endpoint::{EndpointCapabilities, EndpointRef};
pub use error::{CogentError, CogentResult, EndpointError};
#[cfg(feature = "providers")]
pub use error::{ProviderEndpointError, ProviderEndpointErrorKind};
pub use gateway::{
    GatewayAuthentication, GatewayEndpointConfig, GatewayRoutes, GatewaySecret,
    GatewayTimeoutPolicy,
};
#[cfg(feature = "providers")]
pub use provider::{
    AnthropicProviderConfig, OpenAiCompatibleProviderConfig, OpenAiProviderConfig,
    ProviderAuthConfig, ProviderEndpointConfig, ProviderSecret,
};
pub use request::{
    CogentChatRequest, CogentEmbedRequest, CogentQueryRequest, CogentRequestContext,
    CogentTextOptions, EndpointOptions, LocalEmbedOptions, LocalTextOptions, ProviderOptions,
};
pub use response::{CogentEmbeddingResponse, CogentResponseMetadata, CogentTextResponse};
pub use run::{
    CogentCancellationHandle, CogentCancellationReason, CogentEmbeddingResponseFuture,
    CogentEmbeddingRun, CogentTextResponseFuture, CogentTextRun, CogentTokenBatches,
};
