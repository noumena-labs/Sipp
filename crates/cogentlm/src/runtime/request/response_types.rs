use crate::engine::protocol::PoolingType;
use crate::runtime::metrics::RuntimeObservabilityMetrics;

use super::GenerateRequestId;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../../tests/runtime/request/response_types_tests.rs"]
mod response_types_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GenerateResponseStatus {
    #[default]
    Pending = 0,
    Completed,
    Cancelled,
    Failed,
}

impl GenerateResponseStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Completed => "completed",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }
}

/// Internal terminal payload. `query()`/`chat()` finalize with `Text`,
/// `embed()` finalizes with `Embedding`. Public binding mappers enforce the
/// "right command produces the right variant" invariant; the enum itself is
/// never re-exported from `engine::mod`.
#[derive(Debug, Clone, PartialEq)]
pub enum ResponseOutput {
    Text(String),
    Embedding {
        values: Vec<f32>,
        pooling: PoolingType,
        /// Whether the runtime applied L2 normalization to `values`. `Rank`
        /// pooling outputs are never normalized (raw classifier scores).
        normalized: bool,
    },
}

impl Default for ResponseOutput {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenerateResponse {
    pub request_id: GenerateRequestId,
    pub status: GenerateResponseStatus,
    pub output: ResponseOutput,
    pub error_message: String,
    pub runtime_observability: RuntimeObservabilityMetrics,
}

impl GenerateResponse {
    pub fn terminal(
        request_id: GenerateRequestId,
        status: GenerateResponseStatus,
        output: ResponseOutput,
        error_message: impl Into<String>,
    ) -> Self {
        Self {
            request_id,
            status,
            output,
            error_message: error_message.into(),
            ..Self::default()
        }
    }

    pub fn cancelled(request_id: GenerateRequestId, error_message: impl Into<String>) -> Self {
        Self::terminal(
            request_id,
            GenerateResponseStatus::Cancelled,
            ResponseOutput::default(),
            error_message,
        )
    }

    pub fn failed(request_id: GenerateRequestId, error_message: impl Into<String>) -> Self {
        Self::terminal(
            request_id,
            GenerateResponseStatus::Failed,
            ResponseOutput::default(),
            error_message,
        )
    }
}
