//! Lifecycle queue for in-flight generate requests; holds completed responses until the driver consumes them.

use std::collections::{hash_map::Entry, HashMap, VecDeque};
use std::time::Instant;

use super::{
    GenerateRequest, GenerateRequestId, GenerateRequestLifecycle, GenerateResponse,
    GenerateResponseStatus, TokenByteRingProducer,
};
use crate::collection::sorted_copied_values;

#[derive(Debug, Clone)]
pub struct RequestQueue {
    requests: HashMap<GenerateRequestId, GenerateRequest>,
    pending_request_ids: VecDeque<GenerateRequestId>,
    completed_responses: HashMap<GenerateRequestId, GenerateResponse>,
    total_emitted_token_count: i32,
    token_ring_producers: HashMap<GenerateRequestId, TokenByteRingProducer>,
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            requests: HashMap::new(),
            pending_request_ids: VecDeque::new(),
            completed_responses: HashMap::new(),
            total_emitted_token_count: 0,
            token_ring_producers: HashMap::new(),
        }
    }

    pub fn push(&mut self, mut request: GenerateRequest) -> bool {
        let request_id = request.id;
        if request_id == 0 {
            return false;
        }

        request.reset_for_queue();
        request.enqueued_at.get_or_insert_with(Instant::now);
        let Entry::Vacant(entry) = self.requests.entry(request_id) else {
            return false;
        };
        entry.insert(request);
        self.pending_request_ids.push_back(request_id);
        true
    }

    pub fn try_pop_next(&mut self) -> Option<GenerateRequestId> {
        self.try_pop_next_admissible(|_| true)
    }

    pub fn try_pop_next_admissible(
        &mut self,
        predicate: impl Fn(&GenerateRequest) -> bool,
    ) -> Option<GenerateRequestId> {
        let (index, request_id) = self.find_admissible_pending_request(predicate)?;
        self.pending_request_ids.remove(index);
        self.mark_admitted(request_id);
        Some(request_id)
    }

    fn find_admissible_pending_request(
        &self,
        predicate: impl Fn(&GenerateRequest) -> bool,
    ) -> Option<(usize, GenerateRequestId)> {
        self.pending_request_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, request_id)| {
                self.requests.get(request_id).is_some_and(|request| {
                    request.lifecycle == GenerateRequestLifecycle::Pending && predicate(request)
                })
            })
    }

    fn mark_admitted(&mut self, request_id: GenerateRequestId) {
        let Some(request) = self.requests.get_mut(&request_id) else {
            return;
        };
        request.lifecycle = GenerateRequestLifecycle::Admitted;
        request.admitted_at = Some(Instant::now());
    }

    pub fn find_mut(&mut self, request_id: GenerateRequestId) -> Option<&mut GenerateRequest> {
        self.requests.get_mut(&request_id)
    }

    pub fn find(&self, request_id: GenerateRequestId) -> Option<&GenerateRequest> {
        self.requests.get(&request_id)
    }

    pub fn contains(&self, request_id: GenerateRequestId) -> bool {
        self.requests.contains_key(&request_id)
    }

    pub fn cancel(&mut self, request_id: GenerateRequestId, error_message: String) -> bool {
        let Some(request) = self.requests.get_mut(&request_id) else {
            return false;
        };
        let lifecycle = request.lifecycle;
        request.cancel_requested = true;
        let was_pending = lifecycle == GenerateRequestLifecycle::Pending;
        if was_pending {
            self.mark_completed(GenerateResponse::cancelled(request_id, error_message));
        }

        true
    }

    pub fn mark_completed(&mut self, response: GenerateResponse) {
        let request_id = response.request_id;
        self.apply_terminal_response_status(request_id, response.status);

        self.completed_responses.insert(request_id, response);
    }

    pub fn peek_completed_response(
        &self,
        request_id: GenerateRequestId,
    ) -> Option<&GenerateResponse> {
        self.completed_responses.get(&request_id)
    }

    pub fn find_mut_completed_response(
        &mut self,
        request_id: GenerateRequestId,
    ) -> Option<&mut GenerateResponse> {
        self.completed_responses.get_mut(&request_id)
    }

    pub fn completed_response_ids(&self) -> Vec<GenerateRequestId> {
        sorted_copied_values(self.completed_responses.keys().copied())
    }

    pub fn append_streaming_token(&mut self, request_id: GenerateRequestId, text: &str) {
        if request_id == 0 || text.is_empty() {
            return;
        }

        let Some(producer) = self.token_ring_producers.get(&request_id) else {
            return;
        };

        if producer.try_write_frame(request_id, 0, text.as_bytes()) {
            self.total_emitted_token_count = self.total_emitted_token_count.saturating_add(1);
        }
    }

    pub fn add_token_ring_producer(
        &mut self,
        request_id: GenerateRequestId,
        producer: TokenByteRingProducer,
    ) {
        self.token_ring_producers.insert(request_id, producer);
    }

    pub fn remove_token_ring_producer(&mut self, request_id: GenerateRequestId) {
        self.token_ring_producers.remove(&request_id);
    }

    pub fn total_emitted_token_count(&self) -> i32 {
        self.total_emitted_token_count
    }

    pub fn consume_completed_response(&mut self, request_id: GenerateRequestId) -> bool {
        self.take_completed_response(request_id).is_some()
    }

    /// Removes and returns the completed response in one step, avoiding the
    /// peek-then-consume clone path.
    pub fn take_completed_response(
        &mut self,
        request_id: GenerateRequestId,
    ) -> Option<GenerateResponse> {
        let response = self.completed_responses.remove(&request_id)?;
        self.requests.remove(&request_id);
        Some(response)
    }

    pub fn completed_response_count(&self) -> usize {
        self.completed_responses.len()
    }

    pub fn live_request_count(&self) -> usize {
        self.requests
            .len()
            .saturating_sub(self.completed_responses.len())
    }

    pub fn clear(&mut self) {
        self.requests.clear();
        self.pending_request_ids.clear();
        self.completed_responses.clear();
        self.total_emitted_token_count = 0;
        self.token_ring_producers.clear();
    }

    fn remove_pending_request_id(&mut self, request_id: GenerateRequestId) {
        self.pending_request_ids.retain(|&id| id != request_id);
    }

    fn apply_terminal_response_status(
        &mut self,
        request_id: GenerateRequestId,
        status: GenerateResponseStatus,
    ) {
        let Some(request) = self.requests.get_mut(&request_id) else {
            return;
        };
        let was_pending = request.lifecycle == GenerateRequestLifecycle::Pending;
        request.lifecycle =
            GenerateRequestLifecycle::from_response_status(status, request.lifecycle);
        request.completed_at.get_or_insert_with(Instant::now);
        if was_pending {
            self.remove_pending_request_id(request_id);
        }
    }
}

impl GenerateRequestLifecycle {
    fn from_response_status(
        status: GenerateResponseStatus,
        fallback: GenerateRequestLifecycle,
    ) -> Self {
        match status {
            GenerateResponseStatus::Completed => Self::Completed,
            GenerateResponseStatus::Cancelled => Self::Cancelled,
            GenerateResponseStatus::Failed => Self::Failed,
            GenerateResponseStatus::Pending => fallback,
        }
    }
}

#[cfg(test)]
mod tests {
    mod request_queue_tests;
}
