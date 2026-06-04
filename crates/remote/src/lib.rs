//! CogentLM Remote Gateway Protocol client transport.
//!
//! This crate talks only to CogentLM gateways. Provider-specific credentials,
//! upstream URLs, custom headers, and adapter logic belong in gateway/server
//! crates, not in this client transport.

mod config;
mod error;
mod request;
mod response;
mod stream;
mod transport;

pub use cogentlm_core::{CapabilitySupport, TokenUsage};
pub use config::{GatewayConfig, GatewaySecret};
pub use error::{GatewayError, GatewayErrorKind, GatewayResult};
pub use request::{
    GatewayChatRequest, GatewayEmbedRequest, GatewayGenerationOptions, GatewayOptions,
    GatewayQueryRequest,
};
pub use response::{
    GatewayEmbeddingOutput, GatewayEmbeddingResponse, GatewayResponse, GatewayResponseMetadata,
    GatewayTextOutput, GatewayTextResponse,
};
pub use stream::{GatewayStream, GatewayStreamEvent};
pub use transport::GatewayTransport;

#[cfg(test)]
#[path = "tests/transport_tests.rs"]
mod transport_tests;
