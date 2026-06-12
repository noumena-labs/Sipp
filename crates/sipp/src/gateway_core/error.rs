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

impl From<crate::client::SippError> for GatewayError {
    fn from(error: crate::client::SippError) -> Self {
        use crate::client::SippError;

        let kind = match error {
            SippError::Cancelled { .. } => GatewayErrorKind::Cancelled,
            SippError::InvalidRequest(_)
            | SippError::EndpointNotFound(_)
            | SippError::AmbiguousEndpoint { .. }
            | SippError::NoSupportedEndpoint { .. }
            | SippError::UnsupportedOperation { .. } => GatewayErrorKind::InvalidRequest,
            SippError::Internal(_) => GatewayErrorKind::Internal,
            SippError::Local(_) | SippError::Endpoint(_) => GatewayErrorKind::Execution,
            #[cfg(feature = "providers")]
            SippError::Provider(_) => GatewayErrorKind::Execution,
        };
        Self::new(kind, error.to_string())
    }
}
