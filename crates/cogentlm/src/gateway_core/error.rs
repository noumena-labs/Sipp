use thiserror::Error;

/// Result returned by the protocol-neutral gateway pipeline.
pub type GatewayResult<T> = Result<T, GatewayError>;

/// Stable pipeline failure categories without transport semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayErrorKind {
    /// Target resolution failed.
    Resolution,
    /// Authorization denied execution.
    Authorization,
    /// Admission policy rejected execution.
    Admission,
    /// Typed request validation failed.
    InvalidRequest,
    /// Execution failed.
    Execution,
    /// Execution was cancelled.
    Cancelled,
    /// Pipeline infrastructure failed.
    Internal,
}

/// Protocol-neutral gateway pipeline error.
#[derive(Debug, Clone, Error)]
#[error("gateway pipeline error ({kind:?}): {message}")]
pub struct GatewayError {
    /// Stable pipeline category.
    pub kind: GatewayErrorKind,
    /// Human-readable diagnostic.
    pub message: String,
}

impl GatewayError {
    /// Create a pipeline error.
    pub fn new(kind: GatewayErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl From<crate::client::CogentError> for GatewayError {
    fn from(error: crate::client::CogentError) -> Self {
        use crate::client::CogentError;

        let kind = match error {
            CogentError::Cancelled { .. } => GatewayErrorKind::Cancelled,
            CogentError::InvalidRequest(_)
            | CogentError::EndpointNotFound(_)
            | CogentError::AmbiguousEndpoint { .. }
            | CogentError::NoSupportedEndpoint { .. }
            | CogentError::UnsupportedOperation { .. } => GatewayErrorKind::InvalidRequest,
            CogentError::Internal(_) => GatewayErrorKind::Internal,
            CogentError::Local(_) | CogentError::Endpoint(_) => GatewayErrorKind::Execution,
            #[cfg(feature = "providers")]
            CogentError::Provider(_) => GatewayErrorKind::Execution,
        };
        Self::new(kind, error.to_string())
    }
}
