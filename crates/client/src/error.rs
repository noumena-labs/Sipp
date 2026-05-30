use thiserror::Error;

use crate::EndpointRef;

/// Result type used by the unified facade.
pub type CogentResult<T> = Result<T, CogentError>;

/// Error type for endpoint resolution, validation, and endpoint execution.
#[derive(Debug, Error)]
pub enum CogentError {
    #[error(transparent)]
    Local(#[from] cogentlm_engine::Error),

    #[cfg(feature = "providers")]
    #[error(transparent)]
    Provider(#[from] cogentlm_providers::ProviderError),

    #[error("internal facade error: {0}")]
    Internal(String),

    #[error("endpoint not found: {0:?}")]
    EndpointNotFound(EndpointRef),

    #[error("ambiguous endpoint for {operation}")]
    AmbiguousEndpoint { operation: &'static str },

    #[error("no supported endpoint for {operation}")]
    NoSupportedEndpoint { operation: &'static str },

    #[error("unsupported operation {operation} on endpoint {endpoint:?}")]
    UnsupportedOperation {
        endpoint: EndpointRef,
        operation: &'static str,
    },

    #[error("invalid request: {0}")]
    InvalidRequest(String),
}
