use std::collections::{HashMap, VecDeque};
use std::ptr::NonNull;

use cogentlm_sys as ffi;

use crate::runtime::{llama_seq_id, llama_token};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceState {
    pub current_kv_tokens: Vec<llama_token>,
    pub n_past: i32,
    pub hardware_id: llama_seq_id,
    pub pin_count: usize,
}

impl Default for SequenceState {
    fn default() -> Self {
        Self {
            current_kv_tokens: Vec::new(),
            n_past: 0,
            hardware_id: -1,
            pin_count: 0,
        }
    }
}

impl SequenceState {
    pub fn in_sync(&self) -> bool {
        self.n_past == self.current_kv_tokens.len() as i32
    }
}

#[derive(Debug, Clone)]
struct SessionEntry {
    state: SequenceState,
    is_evictable: bool,
}

#[derive(Debug, Clone)]
pub struct SessionStore {
    context_states: HashMap<String, SessionEntry>,
    evictable_context_keys: VecDeque<String>,
    free_seq_ids: VecDeque<llama_seq_id>,
    seq_id_available: Vec<bool>,
    shared_context: Option<NonNull<ffi::llama_context>>,
    max_cached_contexts: usize,
    max_sequences: usize,
}

impl SessionStore {
    pub fn new(max_cached_contexts: usize, max_sequences: usize) -> Self {
        let max_cached_contexts = max_cached_contexts.max(1);
        let max_sequences = max_sequences.max(1);
        let mut free_seq_ids = VecDeque::with_capacity(max_sequences);
        for seq_id in 0..max_sequences {
            free_seq_ids.push_back(seq_id as llama_seq_id);
        }

        Self {
            context_states: HashMap::new(),
            evictable_context_keys: VecDeque::new(),
            free_seq_ids,
            seq_id_available: vec![true; max_sequences],
            shared_context: None,
            max_cached_contexts,
            max_sequences,
        }
    }

    pub fn bind_shared_context(&mut self, shared_context: *mut ffi::llama_context) {
        self.shared_context = NonNull::new(shared_context);
    }

    pub fn find(&self, context_key: &str) -> Option<&SequenceState> {
        self.context_states
            .get(context_key)
            .map(|entry| &entry.state)
    }

    pub fn find_mut(&mut self, context_key: &str) -> Option<&mut SequenceState> {
        self.context_states
            .get_mut(context_key)
            .map(|entry| &mut entry.state)
    }

    pub fn compute_lcp_reuse(
        &self,
        sequence_state: &SequenceState,
        incoming_tokens: &[llama_token],
    ) -> usize {
        sequence_state
            .current_kv_tokens
            .iter()
            .zip(incoming_tokens.iter())
            .take_while(|(cached, incoming)| cached == incoming)
            .count()
    }

    pub fn can_admit(&self, context_key: &str) -> bool {
        if let Some(existing) = self.find(context_key) {
            return existing.pin_count == 0;
        }

        let needs_cache_slot = self.context_states.len() >= self.max_cached_contexts;
        let needs_sequence = self.free_seq_ids.is_empty();
        if !needs_cache_slot && !needs_sequence {
            return true;
        }

        self.has_evictable_session()
    }

    pub fn get_or_create_session(&mut self, context_key: impl Into<String>) -> &mut SequenceState {
        let context_key = context_key.into();
        if self.context_states.contains_key(&context_key) {
            self.touch(&context_key);
            return self
                .context_states
                .get_mut(&context_key)
                .map(|entry| &mut entry.state)
                .expect("existing session");
        }

        self.enforce_limit_before_insert();
        self.emplace(context_key, SequenceState::default())
    }

    pub fn emplace(
        &mut self,
        context_key: impl Into<String>,
        state: SequenceState,
    ) -> &mut SequenceState {
        let context_key = context_key.into();
        self.context_states.insert(
            context_key.clone(),
            SessionEntry {
                state,
                is_evictable: false,
            },
        );
        self.mark_evictable(&context_key);
        self.context_states
            .get_mut(&context_key)
            .map(|entry| &mut entry.state)
            .expect("inserted session")
    }

    pub fn touch(&mut self, context_key: &str) {
        if self
            .context_states
            .get(context_key)
            .is_some_and(|entry| entry.is_evictable)
        {
            remove_key(&mut self.evictable_context_keys, context_key);
            self.evictable_context_keys
                .push_back(context_key.to_string());
        }
    }

    pub fn pin(&mut self, context_key: &str) {
        let Some(entry) = self.context_states.get_mut(context_key) else {
            return;
        };
        entry.state.pin_count += 1;
        self.mark_pinned(context_key);
    }

    pub fn unpin(&mut self, context_key: &str) {
        let Some(entry) = self.context_states.get_mut(context_key) else {
            return;
        };
        entry.state.pin_count = entry.state.pin_count.saturating_sub(1);
        if entry.state.pin_count == 0 {
            self.mark_evictable(context_key);
        }
    }

    pub fn remove(&mut self, context_key: &str) {
        if self.context_states.remove(context_key).is_some() {
            remove_key(&mut self.evictable_context_keys, context_key);
        }
    }

    pub fn enforce_limit_before_insert(&mut self) {
        while (self.context_states.len() >= self.max_cached_contexts
            || self.free_seq_ids.is_empty())
            && !self.evictable_context_keys.is_empty()
        {
            let evict_key = self
                .evictable_context_keys
                .pop_front()
                .expect("evictable key");
            self.context_states.remove(&evict_key);
        }
    }

    pub fn clear(&mut self) {
        self.context_states.clear();
        self.evictable_context_keys.clear();
    }

    pub fn clear_sequence_memory(&self, seq_id: llama_seq_id) {
        let Some(shared_context) = self.shared_context else {
            return;
        };
        if seq_id < 0 {
            return;
        }

        unsafe {
            let memory = ffi::llama_get_memory(shared_context.as_ptr());
            ffi::llama_memory_seq_rm(memory, seq_id, 0, -1);
        }
    }

    pub fn acquire_seq_id(&mut self, hint: llama_seq_id) -> llama_seq_id {
        if self.free_seq_ids.is_empty() {
            return -1;
        }

        let mut seq_id = -1;
        if hint >= 0 && (hint as usize) < self.seq_id_available.len() {
            if let Some(index) = self
                .free_seq_ids
                .iter()
                .position(|candidate| *candidate == hint)
            {
                seq_id = hint;
                self.free_seq_ids.remove(index);
            }
        }

        if seq_id == -1 {
            seq_id = self.free_seq_ids.pop_front().expect("free seq id");
        }

        if seq_id >= 0 && (seq_id as usize) < self.seq_id_available.len() {
            self.seq_id_available[seq_id as usize] = false;
        }

        seq_id
    }

    pub fn release_seq_id(&mut self, seq_id: llama_seq_id) {
        if seq_id < 0 || (seq_id as usize) >= self.seq_id_available.len() {
            return;
        }
        if self.seq_id_available[seq_id as usize] {
            return;
        }

        self.seq_id_available[seq_id as usize] = true;
        // Prefer reusing the most recently released sequence for serial
        // requests. This keeps one-request-at-a-time browser runs on the warm
        // physical KV sequence instead of alternating across n_parallel slots.
        self.free_seq_ids.push_front(seq_id);
    }

    pub fn len(&self) -> usize {
        self.context_states.len()
    }

    pub fn max_sequences(&self) -> usize {
        self.max_sequences
    }

    fn mark_evictable(&mut self, context_key: &str) {
        let Some(entry) = self.context_states.get_mut(context_key) else {
            return;
        };
        if entry.state.pin_count > 0 {
            self.mark_pinned(context_key);
            return;
        }
        if entry.is_evictable {
            remove_key(&mut self.evictable_context_keys, context_key);
        }

        self.evictable_context_keys
            .push_back(context_key.to_string());
        entry.is_evictable = true;
    }

    fn mark_pinned(&mut self, context_key: &str) {
        let Some(entry) = self.context_states.get_mut(context_key) else {
            return;
        };
        if !entry.is_evictable {
            return;
        }

        remove_key(&mut self.evictable_context_keys, context_key);
        entry.is_evictable = false;
    }

    fn has_evictable_session(&self) -> bool {
        !self.evictable_context_keys.is_empty()
    }
}

impl Default for SessionStore {
    fn default() -> Self {
        Self::new(8, 1)
    }
}

fn remove_key(keys: &mut VecDeque<String>, context_key: &str) {
    if let Some(index) = keys.iter().position(|candidate| candidate == context_key) {
        keys.remove(index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_state_reports_sync_with_kv_length() {
        let mut state = SequenceState {
            current_kv_tokens: vec![1, 2, 3],
            n_past: 3,
            ..SequenceState::default()
        };
        assert!(state.in_sync());

        state.n_past = 2;
        assert!(!state.in_sync());
    }

    #[test]
    fn computes_longest_common_prefix() {
        let store = SessionStore::default();
        let state = SequenceState {
            current_kv_tokens: vec![1, 2, 3, 4],
            n_past: 4,
            ..SequenceState::default()
        };

        assert_eq!(store.compute_lcp_reuse(&state, &[1, 2, 9]), 2);
        assert_eq!(store.compute_lcp_reuse(&state, &[9]), 0);
    }

    #[test]
    fn evicts_oldest_evictable_session_before_insert() {
        let mut store = SessionStore::new(2, 2);
        store.emplace("a", SequenceState::default());
        store.emplace("b", SequenceState::default());
        store.touch("a");

        store.get_or_create_session("c");

        assert!(store.find("a").is_some());
        assert!(store.find("b").is_none());
        assert!(store.find("c").is_some());
    }

    #[test]
    fn pinned_session_blocks_admission_when_no_evictable_slot_exists() {
        let mut store = SessionStore::new(1, 1);
        store.emplace("a", SequenceState::default());
        store.pin("a");

        assert!(!store.can_admit("b"));
        store.unpin("a");
        assert!(store.can_admit("b"));
    }

    #[test]
    fn acquire_seq_id_honors_available_hint() {
        let mut store = SessionStore::new(2, 3);

        assert_eq!(store.acquire_seq_id(2), 2);
        assert_eq!(store.acquire_seq_id(2), 0);
        store.release_seq_id(2);
        assert_eq!(store.acquire_seq_id(2), 2);
    }

    #[test]
    fn release_seq_id_reuses_recent_sequence_first() {
        let mut store = SessionStore::new(2, 3);

        assert_eq!(store.acquire_seq_id(-1), 0);
        assert_eq!(store.acquire_seq_id(-1), 1);
        store.release_seq_id(0);

        assert_eq!(store.acquire_seq_id(-1), 0);
    }
}
