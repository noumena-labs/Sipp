use crate::runtime::request::{GenerateRequestId, GenerateResponse};
use crate::runtime::REQUEST_CANCELLED_MESSAGE;

use super::super::InferenceRuntime;

impl InferenceRuntime {
    pub fn cancel_request(&mut self, request_id: GenerateRequestId) -> bool {
        if request_id == 0 {
            return false;
        }
        let mut cancelled = self
            .request_queue
            .cancel(request_id, REQUEST_CANCELLED_MESSAGE.to_string());

        for slot in &mut self.slot_scheduler.slots {
            if slot.request_id != request_id {
                continue;
            }
            if let Some(request) = slot.request_mut() {
                request.cancel_requested = true;
                cancelled = true;
            }
        }

        cancelled
    }

    /// Removes the completed response and its request bookkeeping.
    pub fn take_completed_response(
        &mut self,
        request_id: GenerateRequestId,
    ) -> Option<GenerateResponse> {
        self.committed_observability_request_ids.remove(&request_id);
        self.request_queue.take_completed_response(request_id)
    }
}
