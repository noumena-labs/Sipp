use std::collections::{HashMap, VecDeque};
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
    token_ring_producer: Option<TokenByteRingProducer>,
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
            token_ring_producer: None,
        }
    }

    pub fn push(&mut self, mut request: GenerateRequest) -> bool {
        let request_id = request.id;
        if request_id == 0 || self.requests.contains_key(&request_id) {
            return false;
        }

        request.reset_for_queue();
        request.enqueued_at.get_or_insert_with(Instant::now);
        self.requests.insert(request_id, request);
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
        let mut index = 0;
        while index < self.pending_request_ids.len() {
            let request_id = self.pending_request_ids[index];
            let Some(request) = self.requests.get(&request_id) else {
                self.pending_request_ids.remove(index);
                continue;
            };

            if !predicate(request) {
                index += 1;
                continue;
            }

            self.pending_request_ids.remove(index);
            if let Some(request) = self.requests.get_mut(&request_id) {
                request.lifecycle = GenerateRequestLifecycle::Admitted;
                request.admitted_at = Some(Instant::now());
            }
            return Some(request_id);
        }

        None
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
        let Some(lifecycle) = self
            .requests
            .get(&request_id)
            .map(|request| request.lifecycle)
        else {
            return false;
        };

        if lifecycle == GenerateRequestLifecycle::Pending {
            self.remove_pending_request_id(request_id);
        }

        let request = self
            .requests
            .get_mut(&request_id)
            .expect("request exists after lifecycle check");
        request.cancel_requested = true;
        if lifecycle == GenerateRequestLifecycle::Pending {
            request.lifecycle = GenerateRequestLifecycle::Cancelled;
            request.completed_at = Some(Instant::now());

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
        if self.requests.contains_key(&response.request_id) {
            let request_id = response.request_id;
            self.remove_pending_request_id(request_id);
            let request = self
                .requests
                .get_mut(&request_id)
                .expect("request exists after contains check");
            request.lifecycle = match response.status {
                GenerateResponseStatus::Completed => GenerateRequestLifecycle::Completed,
                GenerateResponseStatus::Cancelled => GenerateRequestLifecycle::Cancelled,
                GenerateResponseStatus::Failed => GenerateRequestLifecycle::Failed,
                GenerateResponseStatus::Pending => request.lifecycle,
            };
            request.completed_at.get_or_insert_with(Instant::now);
        }

        self.completed_responses
            .insert(response.request_id, response);
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
        self.completed_responses.keys().copied().collect()
    }

    pub fn append_streaming_token(&mut self, request_id: GenerateRequestId, text: &str) {
        if request_id == 0 || text.is_empty() {
            return;
        }

        let Some(producer) = &self.token_ring_producer else {
            return;
        };

        if producer.try_write_frame(request_id, 0, text.as_bytes()) {
            self.total_emitted_token_count += 1;
        }
    }

    pub fn set_token_ring_producer(&mut self, producer: Option<TokenByteRingProducer>) {
        self.token_ring_producer = producer;
    }

    pub fn total_emitted_token_count(&self) -> i32 {
        self.total_emitted_token_count
    }

    pub fn consume_completed_response(&mut self, request_id: GenerateRequestId) -> bool {
        if self.completed_responses.remove(&request_id).is_none() {
            return false;
        }

        self.remove_pending_request_id(request_id);
        self.requests.remove(&request_id);
        true
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
        self.token_ring_producer = None;
    }

    fn remove_pending_request_id(&mut self, request_id: GenerateRequestId) {
        if let Some(index) = self
            .pending_request_ids
            .iter()
            .position(|pending_request_id| *pending_request_id == request_id)
        {
            self.pending_request_ids.remove(index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(id: GenerateRequestId) -> GenerateRequest {
        GenerateRequest::new(id, format!("ctx-{id}"))
    }

    #[test]
    fn rejects_zero_and_duplicate_request_ids() {
        let mut queue = RequestQueue::new();
        assert!(!queue.push(request(0)));
        assert!(queue.push(request(1)));
        assert!(!queue.push(request(1)));
    }

    #[test]
    fn pops_first_admissible_request_and_marks_admitted() {
        let mut queue = RequestQueue::new();
        assert!(queue.push(request(1)));
        assert!(queue.push(request(2)));

        let popped = queue.try_pop_next_admissible(|request| request.id == 2);
        assert_eq!(popped, Some(2));
        assert_eq!(
            queue.find(2).map(|request| request.lifecycle),
            Some(GenerateRequestLifecycle::Admitted)
        );
        assert_eq!(queue.try_pop_next(), Some(1));
    }

    #[test]
    fn cancelling_pending_request_creates_completed_response() {
        let mut queue = RequestQueue::new();
        assert!(queue.push(request(7)));
        assert!(queue.cancel(7, "cancelled".to_string()));
        assert_eq!(queue.try_pop_next(), None);

        let response = queue.peek_completed_response(7).expect("response");
        assert_eq!(response.status, GenerateResponseStatus::Cancelled);
        assert_eq!(response.error_message, "cancelled");
    }

    #[test]
    fn cancelling_admitted_request_marks_it_for_runtime_cancellation() {
        let mut queue = RequestQueue::new();
        assert!(queue.push(request(8)));
        assert_eq!(queue.try_pop_next(), Some(8));

        assert!(queue.cancel(8, "cancelled".to_string()));

        let request = queue.find(8).expect("admitted request");
        assert!(request.cancel_requested);
        assert_eq!(request.lifecycle, GenerateRequestLifecycle::Admitted);
        assert!(queue.peek_completed_response(8).is_none());
    }

    #[test]
    fn append_streaming_token_without_ring_is_a_noop() {
        let mut queue = RequestQueue::new();
        queue.append_streaming_token(1, "a");

        assert_eq!(queue.total_emitted_token_count(), 0);
    }

    #[test]
    fn append_streaming_token_writes_to_token_ring() {
        let mut queue = RequestQueue::new();
        let (producer, consumer) = super::super::token_byte_ring(1024);
        queue.set_token_ring_producer(Some(producer));

        queue.append_streaming_token(9, "tok");

        let drain = consumer.drain_available(16, 1024);
        assert_eq!(drain.frames.len(), 1);
        assert_eq!(drain.frames[0].stream_id, 9);
        assert_eq!(drain.frames[0].bytes, b"tok");
        assert_eq!(queue.total_emitted_token_count(), 1);
    }
}
