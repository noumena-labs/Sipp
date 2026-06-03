//! Unified local and remote inference facade.
//!
//! The crate owns endpoint resolution and shared request/result envelopes.
//! Local runtime work stays in `cogentlm-engine`; remote transport execution
//! stays behind the optional `remote` feature.

mod client;
mod dispatch;
mod endpoint;
mod error;
mod local_endpoint;
mod map;
#[cfg(feature = "remote")]
mod remote;
#[cfg(feature = "remote")]
mod remote_endpoint;
#[cfg(feature = "remote")]
mod remote_executor;
mod request;
mod response;
mod run;
mod validate;

pub use client::CogentClient;
pub use endpoint::{EndpointCapabilities, EndpointRef};
pub use error::{CogentError, CogentResult};
#[cfg(feature = "remote")]
pub use error::{RemoteError, RemoteErrorKind};
#[cfg(feature = "remote")]
pub use remote::{RemoteGatewayConfig, RemoteSecret};
pub use request::{
    CogentChatRequest, CogentEmbedRequest, CogentQueryRequest, CogentTextOptions, GatewayOptions,
    LocalEmbedOptions, LocalTextOptions,
};
pub use response::{CogentEmbeddingResponse, CogentTextResponse};
pub use run::{
    CogentEmbeddingResponseFuture, CogentEmbeddingRun, CogentTextResponseFuture, CogentTextRun,
    CogentTokenBatches,
};
