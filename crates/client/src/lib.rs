//! Unified local-engine and provider-facing inference facade.
//!
//! The crate owns endpoint resolution and shared request/result envelopes.
//! Local runtime work stays in `cogentlm-engine`; remote provider execution
//! stays in `cogentlm-providers` behind the optional `providers` feature.

mod client;
mod dispatch;
mod endpoint;
mod engine_endpoint;
mod error;
#[cfg(feature = "providers")]
mod executor;
mod map;
#[cfg(feature = "providers")]
mod provider_endpoint;
mod request;
mod response;
mod run;
mod validate;

pub use client::CogentClient;
pub use endpoint::{EndpointCapabilities, EndpointRef};
pub use error::{CogentError, CogentResult};
#[cfg(feature = "providers")]
pub use executor::ProviderExecutor;
pub use request::{
    CogentChatRequest, CogentEmbedRequest, CogentQueryRequest, CogentTextOptions,
    LocalEmbedOptions, LocalTextOptions, ProviderOptions,
};
pub use response::{CogentEmbeddingResponse, CogentTextResponse};
pub use run::{
    CogentEmbeddingResponseFuture, CogentEmbeddingRun, CogentTextResponseFuture, CogentTextRun,
    CogentTokenStream,
};
