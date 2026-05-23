use crate::runtime::metrics::RuntimeObservabilityMetrics;

use super::GenerateRequestId;

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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenerateResponse {
    pub request_id: GenerateRequestId,
    pub status: GenerateResponseStatus,
    pub output_text: String,
    pub error_message: String,
    pub runtime_observability: RuntimeObservabilityMetrics,
}

impl GenerateResponse {
    pub fn terminal(
        request_id: GenerateRequestId,
        status: GenerateResponseStatus,
        output_text: impl Into<String>,
        error_message: impl Into<String>,
    ) -> Self {
        Self {
            request_id,
            status,
            output_text: output_text.into(),
            error_message: error_message.into(),
            ..Self::default()
        }
    }

    pub fn completed(request_id: GenerateRequestId, output_text: impl Into<String>) -> Self {
        Self::terminal(
            request_id,
            GenerateResponseStatus::Completed,
            output_text,
            "",
        )
    }

    pub fn cancelled(request_id: GenerateRequestId, error_message: impl Into<String>) -> Self {
        Self::terminal(
            request_id,
            GenerateResponseStatus::Cancelled,
            "",
            error_message,
        )
    }

    pub fn failed(request_id: GenerateRequestId, error_message: impl Into<String>) -> Self {
        Self::terminal(
            request_id,
            GenerateResponseStatus::Failed,
            "",
            error_message,
        )
    }
}
