//! Unified local and remote inference facade.
//!
//! The crate owns endpoint resolution and shared request/result envelopes.
//! Local runtime work stays in `cogentlm-engine`; remote transport execution
//! stays behind the optional `providers` feature.

mod client;
mod dispatch;
mod endpoint;
mod error;
mod local_endpoint;
mod map;
#[cfg(feature = "providers")]
mod remote;
#[cfg(feature = "providers")]
mod remote_endpoint;
#[cfg(feature = "providers")]
mod remote_executor;
mod request;
mod response;
mod run;
mod validate;

pub use client::CogentClient;
pub use endpoint::{EndpointCapabilities, EndpointRef};
pub use error::{CogentError, CogentResult};
#[cfg(feature = "providers")]
pub use error::{RemoteError, RemoteErrorKind, RemoteKind};
#[cfg(feature = "providers")]
pub use remote::{
    RemoteAnthropicConfig, RemoteAuth, RemoteConfig, RemoteOpenAiConfig, RemoteProtocol,
    RemoteProxyConfig, RemoteSecret,
};
pub use request::{
    CogentChatRequest, CogentEmbedRequest, CogentQueryRequest, CogentTextOptions,
    LocalEmbedOptions, LocalTextOptions, RemoteOptions,
};
pub use response::{CogentEmbeddingResponse, CogentTextResponse};
pub use run::{
    CogentEmbeddingResponseFuture, CogentEmbeddingRun, CogentTextResponseFuture, CogentTextRun,
    CogentTokenBatches,
};
