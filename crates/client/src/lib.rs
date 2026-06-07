//! Unified local, gateway, and provider inference facade.
//!
//! The crate owns endpoint resolution and shared request/result envelopes.
//! Local runtime work stays in `cogentlm-engine`; gateway transport execution
//! stays behind the optional `remote` feature; direct provider execution stays
//! behind the optional `providers` feature.

mod client;
mod descriptor;
mod dispatch;
mod endpoint;
mod error;
mod local_endpoint;
mod map;
#[cfg(feature = "providers")]
mod provider;
#[cfg(feature = "providers")]
mod provider_endpoint;
#[cfg(feature = "remote")]
mod remote;
#[cfg(feature = "remote")]
mod remote_endpoint;
#[cfg(any(feature = "remote", feature = "providers"))]
mod remote_executor;
mod request;
mod response;
mod run;
mod validate;

pub use client::CogentClient;
pub use descriptor::{EndpointDescriptor, LocalModelDescriptor};
pub use endpoint::{EndpointCapabilities, EndpointRef};
pub use error::{CogentError, CogentResult};
#[cfg(feature = "providers")]
pub use error::{ProviderEndpointError, ProviderEndpointErrorKind};
#[cfg(feature = "remote")]
pub use error::{RemoteError, RemoteErrorKind};
#[cfg(feature = "providers")]
pub use provider::{
    AnthropicProviderConfig, OpenAiCompatibleProviderConfig, OpenAiProviderConfig,
    ProviderAuthConfig, ProviderEndpointConfig, ProviderSecret,
};
#[cfg(feature = "remote")]
pub use remote::{RemoteGatewayConfig, RemoteSecret};
pub use request::{
    CogentChatRequest, CogentEmbedRequest, CogentQueryRequest, CogentRequestContext,
    CogentTextOptions, GatewayOptions, LocalEmbedOptions, LocalTextOptions, ProviderOptions,
};
pub use response::{CogentEmbeddingResponse, CogentResponseMetadata, CogentTextResponse};
pub use run::{
    CogentCancellationHandle, CogentCancellationReason, CogentEmbeddingResponseFuture,
    CogentEmbeddingRun, CogentTextResponseFuture, CogentTextRun, CogentTokenBatches,
};
