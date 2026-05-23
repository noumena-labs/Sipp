//! Per-context-key session map: tracks live KV tokens, hardware sequence ids, and eviction.

use std::collections::{hash_map::Entry, HashMap, VecDeque};
use std::ptr::NonNull;

use cogentlm_sys as ffi;

use crate::runtime::{llama_seq_id, llama_token};

mod sequence_ids;
mod state;

use sequence_ids::clamp_sequence_capacity;
pub use state::SequenceState;

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
        let max_sequences = clamp_sequence_capacity(max_sequences);
        let mut free_seq_ids = VecDeque::with_capacity(max_sequences);
        for seq_id in 0..max_sequences {
            let Ok(seq_id) = llama_seq_id::try_from(seq_id) else {
                break;
            };
            free_seq_ids.push_back(seq_id);
        }

        Self {
            context_states: HashMap::with_capacity(max_cached_contexts),
            evictable_context_keys: VecDeque::with_capacity(max_cached_contexts),
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
        let has_evictable = self.has_evictable_session();
        self.can_admit_with_evictable_cached(context_key, has_evictable)
    }

    pub fn can_admit_with_evictable_cached(&self, context_key: &str, has_evictable: bool) -> bool {
        if self.free_seq_ids.is_empty() {
            return false;
        }

        if let Some(existing) = self.find(context_key) {
            return existing.pin_count == 0;
        }

        let needs_cache_slot = self.context_states.len() >= self.max_cached_contexts;
        if !needs_cache_slot {
            return true;
        }

        has_evictable
    }

    pub fn get_or_create_session(&mut self, context_key: &str) -> Option<&mut SequenceState> {
        if self.context_states.contains_key(context_key) {
            self.touch(context_key);
            return self.find_mut(context_key);
        }

        self.enforce_limit_before_insert();
        Some(self.emplace(context_key.to_string(), SequenceState::default()))
    }

    pub fn emplace(
        &mut self,
        context_key: impl Into<String>,
        state: SequenceState,
    ) -> &mut SequenceState {
        let context_key = context_key.into();
        refresh_evictable_key(&mut self.evictable_context_keys, context_key.clone());
        let entry = match self.context_states.entry(context_key) {
            Entry::Occupied(mut occupied) => {
                occupied.insert(SessionEntry {
                    state,
                    is_evictable: true,
                });
                occupied.into_mut()
            }
            Entry::Vacant(vacant) => vacant.insert(SessionEntry {
                state,
                is_evictable: true,
            }),
        };
        &mut entry.state
    }

    pub fn touch(&mut self, context_key: &str) {
        if self
            .context_states
            .get(context_key)
            .is_some_and(|entry| entry.is_evictable)
        {
            refresh_evictable_key(&mut self.evictable_context_keys, context_key.to_string());
        }
    }

    pub fn pin(&mut self, context_key: &str) {
        let Some(entry) = self.context_states.get_mut(context_key) else {
            return;
        };
        entry.state.pin_count = entry.state.pin_count.saturating_add(1);
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
            let Some(evict_key) = self.evictable_context_keys.pop_front() else {
                break;
            };
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

        // SAFETY: `shared_context` is captured from `bind_shared_context` as a
        // non-null llama context pointer owned by the runtime. `seq_id` is
        // validated non-negative before passing it to llama.cpp.
        unsafe {
            let memory = ffi::llama_get_memory(shared_context.as_ptr());
            ffi::llama_memory_seq_rm(memory, seq_id, 0, -1);
        }
    }

    pub fn prepare_for_admission(
        &mut self,
        context_key: &str,
        leased_seq_id: llama_seq_id,
    ) -> Option<SequenceState> {
        let session = self.find_mut(context_key)?;
        if leased_seq_id != session.hardware_id {
            session.current_kv_tokens.clear();
            session.n_past = 0;
        }
        session.hardware_id = leased_seq_id;

        let tokens = std::mem::take(&mut session.current_kv_tokens);
        let mut snapshot = session.clone();
        snapshot.current_kv_tokens = tokens;
        Some(snapshot)
    }

    pub fn len(&self) -> usize {
        self.context_states.len()
    }

    pub fn is_empty(&self) -> bool {
        self.context_states.is_empty()
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
        refresh_evictable_key(&mut self.evictable_context_keys, context_key.to_string());
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

    pub(crate) fn has_evictable_session(&self) -> bool {
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

fn refresh_evictable_key(keys: &mut VecDeque<String>, context_key: String) {
    remove_key(keys, &context_key);
    keys.push_back(context_key);
}

#[cfg(test)]
mod tests {
    mod session_store_tests;
}
