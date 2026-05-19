//! Lifecycle queue for in-flight generate requests; holds completed responses until the driver consumes them.

use std::collections::{hash_map::Entry, HashMap, VecDeque};
use std::time::Instant;

use super::{
    GenerateRequest, GenerateRequestId, GenerateRequestLifecycle, GenerateResponse,
    GenerateResponseStatus, TokenByteRingProducer,
};

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
        let mut found_index = None;
        for (index, &request_id) in self.pending_request_ids.iter().enumerate() {
            let Some(request) = self.requests.get(&request_id) else {
                continue;
            };
            if request.lifecycle != GenerateRequestLifecycle::Pending {
                continue;
            }
            if predicate(request) {
                found_index = Some((index, request_id));
                break;
            }
        }

        if let Some((index, request_id)) = found_index {
            self.pending_request_ids.remove(index);
            if let Some(request) = self.requests.get_mut(&request_id) {
                request.lifecycle = GenerateRequestLifecycle::Admitted;
                request.admitted_at = Some(Instant::now());
            }
            Some(request_id)
        } else {
            None
        }
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
            request.lifecycle = GenerateRequestLifecycle::Cancelled;
            request.completed_at.get_or_insert_with(Instant::now);
        }

        if was_pending {
            self.remove_pending_request_id(request_id);
            self.completed_responses.insert(
                request_id,
                GenerateResponse {
                    request_id,
                    status: GenerateResponseStatus::Cancelled,
                    error_message,
                    ..GenerateResponse::default()
                },
            );
        }

        true
    }

    pub fn mark_completed(&mut self, response: GenerateResponse) {
        let request_id = response.request_id;
        if let Some(request) = self.requests.get_mut(&request_id) {
            let was_pending = request.lifecycle == GenerateRequestLifecycle::Pending;
            request.lifecycle = match response.status {
                GenerateResponseStatus::Completed => GenerateRequestLifecycle::Completed,
                GenerateResponseStatus::Cancelled => GenerateRequestLifecycle::Cancelled,
                GenerateResponseStatus::Failed => GenerateRequestLifecycle::Failed,
                GenerateResponseStatus::Pending => request.lifecycle,
            };
            request.completed_at.get_or_insert_with(Instant::now);
            if was_pending {
                self.remove_pending_request_id(request_id);
            }
        }

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
        let mut ids = Vec::with_capacity(self.completed_responses.len());
        ids.extend(self.completed_responses.keys().copied());
        ids.sort_unstable();
        ids
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
        if self.completed_responses.remove(&request_id).is_none() {
            return false;
        }

        self.requests.remove(&request_id);
        true
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
        if self.completed_responses.len() > self.requests.len() {
            0
        } else {
            self.requests.len() - self.completed_responses.len()
        }
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
}

#[cfg(test)]
mod tests;
