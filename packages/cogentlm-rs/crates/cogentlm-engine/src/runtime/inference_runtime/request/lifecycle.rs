use crate::runtime::request::{GenerateRequestId, GenerateResponse, TokenByteRingProducer};
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

        for slot in self.slot_scheduler.mutable_slots() {
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

    pub fn try_peek_completed_response(
        &self,
        request_id: GenerateRequestId,
    ) -> Option<GenerateResponse> {
        self.request_queue
            .peek_completed_response(request_id)
            .cloned()
    }

    /// Removes the completed response and its request bookkeeping.
    pub fn take_completed_response(
        &mut self,
        request_id: GenerateRequestId,
    ) -> Option<GenerateResponse> {
        self.committed_observability_request_ids.remove(&request_id);
        self.request_queue.take_completed_response(request_id)
    }

    pub fn has_request(&self, request_id: GenerateRequestId) -> bool {
        self.request_queue.contains(request_id)
    }

    pub fn consume_completed_response(&mut self, request_id: GenerateRequestId) -> bool {
        self.committed_observability_request_ids.remove(&request_id);
        self.request_queue.consume_completed_response(request_id)
    }

    pub fn add_token_ring_producer(
        &mut self,
        request_id: GenerateRequestId,
        producer: TokenByteRingProducer,
    ) {
        self.request_queue
            .add_token_ring_producer(request_id, producer);
    }

    pub fn remove_token_ring_producer(&mut self, request_id: GenerateRequestId) {
        self.request_queue.remove_token_ring_producer(request_id);
    }
}
