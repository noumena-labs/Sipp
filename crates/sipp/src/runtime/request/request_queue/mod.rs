//! Lifecycle queue for in-flight generate requests; holds completed responses until the driver consumes them.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use super::{
    GenerateRequest, GenerateRequestId, GenerateRequestLifecycle, GenerateResponse,
    GenerateResponseStatus, TokenEmissionSinkRef,
};
#[derive(Debug, Clone)]
pub struct RequestQueue {
    entries: HashMap<GenerateRequestId, RequestQueueEntry>,
    pending_request_ids: VecDeque<GenerateRequestId>,
    pub completed_responses: HashMap<GenerateRequestId, GenerateResponse>,
    pub total_emitted_token_count: i32,
    pub token_emission_sinks: HashMap<GenerateRequestId, TokenEmissionSinkRef>,
    pending_token_emissions: HashMap<GenerateRequestId, PendingTokenEmission>,
}

#[derive(Debug, Clone)]
enum RequestQueueEntry {
    Pending(GenerateRequest),
    State {
        lifecycle: GenerateRequestLifecycle,
        cancel_requested: bool,
    },
}

impl RequestQueueEntry {
    fn lifecycle(&self) -> GenerateRequestLifecycle {
        match self {
            Self::Pending(request) => request.lifecycle,
            Self::State { lifecycle, .. } => *lifecycle,
        }
    }

    fn cancel_requested(&self) -> bool {
        match self {
            Self::Pending(request) => request.cancel_requested,
            Self::State {
                cancel_requested, ..
            } => *cancel_requested,
        }
    }

    fn set_cancel_requested(&mut self) {
        match self {
            Self::Pending(request) => request.cancel_requested = true,
            Self::State {
                cancel_requested, ..
            } => *cancel_requested = true,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct PendingTokenEmission {
    text: String,
    frame_count: u32,
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl RequestQueue {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
            pending_request_ids: VecDeque::new(),
            completed_responses: HashMap::new(),
            total_emitted_token_count: 0,
            token_emission_sinks: HashMap::new(),
            pending_token_emissions: HashMap::new(),
        }
    }

    pub fn push(&mut self, mut request: GenerateRequest) -> bool {
        let request_id = request.id;
        if request_id == 0 {
            return false;
        }

        request.reset_for_queue();
        request.enqueued_at.get_or_insert_with(Instant::now);
        if self.entries.contains_key(&request_id) {
            return false;
        }
        self.entries
            .insert(request_id, RequestQueueEntry::Pending(request));
        self.pending_request_ids.push_back(request_id);
        true
    }

    pub fn try_pop_next_admissible(
        &mut self,
        mut predicate: impl FnMut(&GenerateRequest) -> bool,
    ) -> Option<GenerateRequestId> {
        let (index, request_id) = self.find_admissible_pending_request(&mut predicate)?;
        self.pending_request_ids.remove(index);
        self.mark_admitted(request_id);
        Some(request_id)
    }

    pub(crate) fn take_admitted_request(
        &mut self,
        request_id: GenerateRequestId,
    ) -> Option<GenerateRequest> {
        let entry = self.entries.remove(&request_id)?;
        let RequestQueueEntry::Pending(request) = entry else {
            self.entries.insert(request_id, entry);
            return None;
        };
        let lifecycle = request.lifecycle;
        let cancel_requested = request.cancel_requested;
        self.entries.insert(
            request_id,
            RequestQueueEntry::State {
                lifecycle,
                cancel_requested,
            },
        );
        Some(request)
    }

    pub(crate) fn pending_request(
        &self,
        request_id: GenerateRequestId,
    ) -> Option<&GenerateRequest> {
        match self.entries.get(&request_id) {
            Some(RequestQueueEntry::Pending(request)) => Some(request),
            _ => None,
        }
    }

    pub(crate) fn request_lifecycle(
        &self,
        request_id: GenerateRequestId,
    ) -> Option<GenerateRequestLifecycle> {
        self.entries
            .get(&request_id)
            .map(RequestQueueEntry::lifecycle)
    }

    pub(crate) fn request_cancel_requested(&self, request_id: GenerateRequestId) -> bool {
        self.entries
            .get(&request_id)
            .is_some_and(RequestQueueEntry::cancel_requested)
    }

    pub fn contains_request(&self, request_id: GenerateRequestId) -> bool {
        self.entries.contains_key(&request_id)
    }

    pub fn has_uncompleted_requests(&self) -> bool {
        self.entries.values().any(|entry| {
            !matches!(
                entry.lifecycle(),
                GenerateRequestLifecycle::Completed
                    | GenerateRequestLifecycle::Cancelled
                    | GenerateRequestLifecycle::Failed
            )
        })
    }

    fn find_admissible_pending_request(
        &self,
        predicate: &mut impl FnMut(&GenerateRequest) -> bool,
    ) -> Option<(usize, GenerateRequestId)> {
        self.pending_request_ids
            .iter()
            .copied()
            .enumerate()
            .find(|(_, request_id)| {
                self.pending_request(*request_id).is_some_and(|request| {
                    request.lifecycle == GenerateRequestLifecycle::Pending && predicate(request)
                })
            })
    }

    fn mark_admitted(&mut self, request_id: GenerateRequestId) {
        if let Some(RequestQueueEntry::Pending(request)) = self.entries.get_mut(&request_id) {
            request.lifecycle = GenerateRequestLifecycle::Admitted;
            request.admitted_at = Some(Instant::now());
        }
    }

    pub fn cancel(&mut self, request_id: GenerateRequestId, error_message: String) -> bool {
        let Some(entry) = self.entries.get_mut(&request_id) else {
            return false;
        };
        let lifecycle = entry.lifecycle();
        entry.set_cancel_requested();
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

    pub fn append_token_piece(&mut self, request_id: GenerateRequestId, text: &str) {
        if request_id == 0 || text.is_empty() {
            return;
        }

        if !self.token_emission_sinks.contains_key(&request_id) {
            return;
        };

        let pending = self.pending_token_emissions.entry(request_id).or_default();
        pending.text.push_str(text);
        pending.frame_count = pending.frame_count.saturating_add(1);
        self.total_emitted_token_count = self.total_emitted_token_count.saturating_add(1);
    }

    pub fn has_token_emission_sinks(&self) -> bool {
        !self.token_emission_sinks.is_empty()
    }

    pub fn flush_token_emissions(&mut self) -> bool {
        let mut flushed = false;
        let sinks = &self.token_emission_sinks;
        for (request_id, pending) in self.pending_token_emissions.drain() {
            if pending.text.is_empty() || pending.frame_count == 0 {
                continue;
            }
            let Some(sink) = sinks.get(&request_id) else {
                continue;
            };
            flushed |=
                sink.try_write_batch(request_id, pending.frame_count, pending.text.as_bytes());
        }
        flushed
    }

    /// Removes and returns the completed response in one step, avoiding the
    /// peek-then-consume clone path.
    pub fn take_completed_response(
        &mut self,
        request_id: GenerateRequestId,
    ) -> Option<GenerateResponse> {
        let response = self.completed_responses.remove(&request_id)?;
        self.entries.remove(&request_id);
        self.pending_token_emissions.remove(&request_id);
        Some(response)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.pending_request_ids.clear();
        self.completed_responses.clear();
        self.total_emitted_token_count = 0;
        self.token_emission_sinks.clear();
        self.pending_token_emissions.clear();
    }

    fn remove_pending_request_id(&mut self, request_id: GenerateRequestId) {
        self.pending_request_ids.retain(|&id| id != request_id);
    }

    fn apply_terminal_response_status(
        &mut self,
        request_id: GenerateRequestId,
        status: GenerateResponseStatus,
    ) {
        let Some(entry) = self.entries.get_mut(&request_id) else {
            return;
        };
        let was_pending = matches!(entry, RequestQueueEntry::Pending(_));
        let lifecycle = GenerateRequestLifecycle::from_response_status(status, entry.lifecycle());
        let cancel_requested = entry.cancel_requested();
        *entry = RequestQueueEntry::State {
            lifecycle,
            cancel_requested,
        };
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
#[path = "../../../tests/runtime/request/request_queue_tests.rs"]
mod request_queue_tests;
