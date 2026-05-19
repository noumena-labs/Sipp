//! Unit tests for the parent module.

use super::super::*;

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
fn get_or_create_existing_session_refreshes_eviction_order() {
    let mut store = SessionStore::new(2, 2);
    store.emplace("a", SequenceState::default());
    store.emplace("b", SequenceState::default());

    store.get_or_create_session("a");
    store.get_or_create_session("c");

    assert!(store.find("a").is_some());
    assert!(store.find("b").is_none());
    assert!(store.find("c").is_some());
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
fn emplace_replaces_existing_session_without_duplicate_evictable_keys() {
    let mut store = SessionStore::new(2, 2);
    store.emplace("a", SequenceState::default());
    store.emplace(
        "a",
        SequenceState {
            current_kv_tokens: vec![9],
            n_past: 1,
            ..SequenceState::default()
        },
    );
    store.emplace("b", SequenceState::default());

    store.get_or_create_session("c");

    assert!(store.find("a").is_none());
    assert!(store.find("b").is_some());
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
fn pin_count_saturates_instead_of_wrapping() {
    let mut store = SessionStore::new(1, 1);
    store.emplace(
        "a",
        SequenceState {
            pin_count: usize::MAX,
            ..SequenceState::default()
        },
    );

    store.pin("a");

    assert_eq!(
        store.find("a").map(|state| state.pin_count),
        Some(usize::MAX)
    );
    assert!(!store.can_admit("b"));
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

#[test]
fn sequence_id_bounds_ignore_negative_and_out_of_range_ids() {
    let mut store = SessionStore::new(2, 2);

    assert_eq!(store.acquire_seq_id(-1), 0);
    store.release_seq_id(-1);
    store.release_seq_id(2);

    assert_eq!(store.acquire_seq_id(-1), 1);
    assert_eq!(store.acquire_seq_id(2), -1);
}

#[test]
fn session_store_presizes_bounded_collections() {
    let store = SessionStore::new(3, 4);

    assert!(store.context_states.capacity() >= 3);
    assert!(store.evictable_context_keys.capacity() >= 3);
    assert!(store.free_seq_ids.capacity() >= 4);
}

#[test]
fn prepare_for_admission_moves_tokens_and_resets_on_sequence_change() {
    let mut store = SessionStore::new(2, 2);
    store.emplace(
        "ctx",
        SequenceState {
            current_kv_tokens: vec![1, 2, 3],
            n_past: 3,
            hardware_id: 1,
            pin_count: 0,
        },
    );

    let reused = store.prepare_for_admission("ctx", 1).expect("snapshot");
    assert_eq!(reused.current_kv_tokens, vec![1, 2, 3]);
    assert_eq!(
        store.find("ctx").expect("session").current_kv_tokens,
        Vec::<llama_token>::new()
    );

    store.find_mut("ctx").expect("session").current_kv_tokens = vec![4, 5];
    store.find_mut("ctx").expect("session").n_past = 2;
    let reset = store.prepare_for_admission("ctx", 0).expect("snapshot");
    assert!(reset.current_kv_tokens.is_empty());
    assert_eq!(reset.n_past, 0);
    assert_eq!(reset.hardware_id, 0);
}
