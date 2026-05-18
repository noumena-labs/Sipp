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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct GenerateResponse {
    pub request_id: GenerateRequestId,
    pub status: GenerateResponseStatus,
    pub output_text: String,
    pub error_message: String,
    pub runtime_observability: RuntimeObservabilityMetrics,
}
