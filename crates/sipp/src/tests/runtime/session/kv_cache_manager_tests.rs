//! Tests the `runtime::session::kv_cache_manager` module in `sipp`.
//!
//! Covers runtime support modules with deterministic in-memory fixtures and no native model execution.

use crate::runtime::config::KvReuseMode;

use super::{CacheCandidate, KvCacheManager, ResidentRef, SeqState, SequenceMirror, SessionRecord};

fn mirror(tokens: &[i32]) -> SequenceMirror {
    SequenceMirror {
        current_kv_tokens: tokens.to_vec(),
        n_past: tokens.len() as i32,
    }
}

#[test]
fn never_used_free_sequences_do_not_require_clear() {
    let mut manager = KvCacheManager::new(2);

    let first = manager
        .admit("a", KvReuseMode::LiveSlotPrefix, false)
        .expect("first admission");
    let second = manager
        .admit("b", KvReuseMode::LiveSlotPrefix, false)
        .expect("second admission");

    assert_eq!(first.seq_id, 0);
    assert_eq!(second.seq_id, 1);
    assert!(!first.requires_kv_clear);
    assert!(!second.requires_kv_clear);
}

#[test]
fn warm_reuse_moves_idle_tokens_into_admission_mirror() {
    let mut manager = KvCacheManager::new(1);

    let cold = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("cold admission");
    assert_eq!(cold.candidate, CacheCandidate::None);
    assert_eq!(cold.seq_id, 0);
    assert!(!cold.requires_kv_clear);
    assert!(cold.mirror.current_kv_tokens.is_empty());

    manager.finalize_slot(
        "ctx",
        cold.seq_id,
        cold.lease_epoch,
        mirror(&[1, 2, 3, 4]),
        true,
        KvReuseMode::LiveSlotPrefix,
    );

    let warm = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("warm admission");
    assert_eq!(warm.candidate, CacheCandidate::Live);
    assert_eq!(warm.seq_id, cold.seq_id);
    assert!(warm.lease_epoch > cold.lease_epoch);
    assert!(!warm.requires_kv_clear);
    assert_eq!(warm.mirror.current_kv_tokens, vec![1, 2, 3, 4]);
}

#[test]
fn stale_resident_ref_never_reuses_reassigned_sequence() {
    let mut manager = KvCacheManager::new(1);

    let a = manager
        .admit("a", KvReuseMode::LiveSlotPrefix, false)
        .expect("a admission");
    manager.finalize_slot(
        "a",
        a.seq_id,
        a.lease_epoch,
        mirror(&[1, 2, 3]),
        true,
        KvReuseMode::LiveSlotPrefix,
    );

    let b = manager
        .admit("b", KvReuseMode::LiveSlotPrefix, false)
        .expect("b admission");
    assert_eq!(b.candidate, CacheCandidate::None);
    assert_eq!(b.seq_id, a.seq_id);
    assert!(b.lease_epoch > a.lease_epoch);
    assert!(b.requires_kv_clear);
    manager.finalize_slot(
        "b",
        b.seq_id,
        b.lease_epoch,
        mirror(&[9, 8, 7]),
        true,
        KvReuseMode::LiveSlotPrefix,
    );

    let a_return = manager
        .admit("a", KvReuseMode::LiveSlotPrefix, false)
        .expect("a return admission");
    assert_eq!(a_return.candidate, CacheCandidate::None);
    assert_eq!(a_return.seq_id, a.seq_id);
    assert!(a_return.lease_epoch > b.lease_epoch);
    assert!(a_return.requires_kv_clear);
    assert!(a_return.mirror.current_kv_tokens.is_empty());
}

#[test]
fn disabled_mode_success_evicts_live_residency() {
    let mut manager = KvCacheManager::new(1);

    let admission = manager
        .admit("ctx", KvReuseMode::Disabled, false)
        .expect("admission");
    manager.finalize_slot(
        "ctx",
        admission.seq_id,
        admission.lease_epoch,
        mirror(&[1, 2, 3]),
        true,
        KvReuseMode::Disabled,
    );

    let next = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("next admission");
    assert_eq!(next.candidate, CacheCandidate::None);
    assert!(next.lease_epoch > admission.lease_epoch);
    assert!(next.requires_kv_clear);
    assert!(next.mirror.current_kv_tokens.is_empty());
}

#[test]
fn can_admit_is_false_when_every_sequence_is_leased() {
    let mut manager = KvCacheManager::new(1);

    let _leased = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("admission");

    assert!(!manager.can_admit("other"));
}

#[test]
fn forced_reset_releases_in_flight_state_and_bumps_lease_epoch() {
    let mut manager = KvCacheManager::new(1);

    let admission = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("admission");
    manager.release_slot_for_reset("ctx", admission.seq_id, admission.lease_epoch);

    let next = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("next admission");
    assert_eq!(next.candidate, CacheCandidate::None);
    assert!(next.lease_epoch > admission.lease_epoch);
    assert!(next.requires_kv_clear);
}

#[test]
fn stale_finalize_does_not_overwrite_new_lease() {
    let mut manager = KvCacheManager::new(1);

    let stale = manager
        .admit("stale", KvReuseMode::LiveSlotPrefix, false)
        .expect("stale admission");
    manager.release_slot_for_reset("stale", stale.seq_id, stale.lease_epoch);

    let current = manager
        .admit("current", KvReuseMode::LiveSlotPrefix, false)
        .expect("current admission");
    assert!(current.lease_epoch > stale.lease_epoch);
    assert!(current.requires_kv_clear);

    manager.finalize_slot(
        "stale",
        stale.seq_id,
        stale.lease_epoch,
        mirror(&[1, 2, 3]),
        true,
        KvReuseMode::LiveSlotPrefix,
    );

    let index = usize::try_from(current.seq_id).expect("current seq index");
    assert!(matches!(manager.physical[index].state, SeqState::Leased));
    assert!(!manager.sessions.contains_key("stale"));
    assert!(manager
        .sessions
        .get("current")
        .is_some_and(|record| record.in_flight));
}

#[test]
fn stale_lease_epoch_callbacks_do_not_clear_new_same_context_lease() {
    let mut manager = KvCacheManager::new(1);

    let stale = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("stale admission");
    manager.release_slot_for_reset("ctx", stale.seq_id, stale.lease_epoch);

    let current = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("current admission");
    assert!(current.lease_epoch > stale.lease_epoch);
    assert!(current.requires_kv_clear);

    manager.finalize_slot(
        "ctx",
        stale.seq_id,
        stale.lease_epoch,
        mirror(&[1, 2, 3]),
        true,
        KvReuseMode::LiveSlotPrefix,
    );
    manager.release_slot_for_reset("ctx", stale.seq_id, stale.lease_epoch);

    let index = usize::try_from(current.seq_id).expect("current seq index");
    assert!(matches!(manager.physical[index].state, SeqState::Leased));
    assert!(manager
        .sessions
        .get("ctx")
        .is_some_and(|record| record.in_flight));
}

#[test]
fn bypass_cache_forces_cold_admission_even_with_live_resident() {
    let mut manager = KvCacheManager::new(1);

    let initial = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("initial admission");
    manager.finalize_slot(
        "ctx",
        initial.seq_id,
        initial.lease_epoch,
        mirror(&[1, 2, 3]),
        true,
        KvReuseMode::LiveSlotPrefix,
    );

    let bypass = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, true)
        .expect("bypass admission");
    assert_eq!(bypass.candidate, CacheCandidate::None);
    assert!(bypass.lease_epoch > initial.lease_epoch);
    assert!(bypass.requires_kv_clear);
    assert!(bypass.mirror.current_kv_tokens.is_empty());
}

#[test]
fn lru_scan_drops_stale_residents_and_evicts_only_valid_idle_sequence() {
    let mut manager = KvCacheManager::new(2);

    let stale = manager
        .admit("stale", KvReuseMode::LiveSlotPrefix, false)
        .expect("stale admission");
    manager.finalize_slot(
        "stale",
        stale.seq_id,
        stale.lease_epoch,
        mirror(&[1, 2, 3]),
        true,
        KvReuseMode::LiveSlotPrefix,
    );
    let valid = manager
        .admit("valid", KvReuseMode::LiveSlotPrefix, false)
        .expect("valid admission");
    manager.finalize_slot(
        "valid",
        valid.seq_id,
        valid.lease_epoch,
        mirror(&[4, 5, 6]),
        true,
        KvReuseMode::LiveSlotPrefix,
    );

    manager.physical[usize::try_from(stale.seq_id).expect("stale seq index")].lease_epoch += 1;
    manager.idle_lru.clear();
    manager.idle_lru.push_back("stale".to_string());
    manager.idle_lru.push_back("valid".to_string());
    manager.sessions.insert(
        "dangling".to_string(),
        SessionRecord {
            resident: Some(ResidentRef {
                seq_id: stale.seq_id,
                lease_epoch: stale.lease_epoch,
            }),
            in_flight: false,
        },
    );

    let next = manager
        .admit("next", KvReuseMode::LiveSlotPrefix, false)
        .expect("next admission");

    assert_eq!(next.seq_id, valid.seq_id);
    assert_eq!(next.candidate, CacheCandidate::None);
    assert!(next.requires_kv_clear);
    assert!(manager.sessions.contains_key("next"));
    assert!(!manager.sessions.contains_key("stale"));
    assert!(!manager.sessions.contains_key("valid"));
    assert!(manager.sessions.contains_key("dangling"));
    assert!(matches!(
        manager.physical[usize::try_from(stale.seq_id).expect("stale seq index")].state,
        SeqState::Idle { .. }
    ));
}

#[test]
fn state_snapshot_success_does_not_keep_live_residency() {
    let mut manager = KvCacheManager::new(1);

    let admission = manager
        .admit("ctx", KvReuseMode::StateSnapshot, false)
        .expect("admission");
    manager.finalize_slot(
        "ctx",
        admission.seq_id,
        admission.lease_epoch,
        mirror(&[1, 2, 3]),
        true,
        KvReuseMode::StateSnapshot,
    );

    let next = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("next admission");
    assert_eq!(next.candidate, CacheCandidate::None);
    assert!(next.lease_epoch > admission.lease_epoch);
    assert!(next.requires_kv_clear);
    assert!(next.mirror.current_kv_tokens.is_empty());
}

#[test]
fn failed_slot_evicts_live_residency() {
    let mut manager = KvCacheManager::new(1);

    let admission = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("admission");
    manager.finalize_slot(
        "ctx",
        admission.seq_id,
        admission.lease_epoch,
        mirror(&[1, 2, 3]),
        false,
        KvReuseMode::LiveSlotPrefix,
    );

    let next = manager
        .admit("ctx", KvReuseMode::LiveSlotPrefix, false)
        .expect("next admission");
    assert_eq!(next.candidate, CacheCandidate::None);
    assert!(next.lease_epoch > admission.lease_epoch);
    assert!(next.requires_kv_clear);
}

#[test]
fn queued_snapshot_is_dropped_when_sequence_is_released() {
    let mut manager = KvCacheManager::new(1);

    let admission = manager
        .admit("ctx", KvReuseMode::LiveSlotAndSnapshot, false)
        .expect("admission");

    assert!(manager.queue_prefix_snapshot(
        7,
        "ctx",
        admission.seq_id,
        admission.lease_epoch,
        &[1, 2],
        2,
    ));
    assert_eq!(manager.pending_prefix_snapshot_count(), 1);

    manager.release_slot_for_reset("ctx", admission.seq_id, admission.lease_epoch);

    assert_eq!(manager.pending_prefix_snapshot_count(), 0);
}

#[test]
fn warm_reuse_invalidates_pending_snapshot_and_advances_lease_epoch() {
    let mut manager = KvCacheManager::new(1);

    let admission = manager
        .admit("ctx", KvReuseMode::LiveSlotAndSnapshot, false)
        .expect("admission");
    assert!(manager.queue_prefix_snapshot(
        7,
        "ctx",
        admission.seq_id,
        admission.lease_epoch,
        &[1, 2],
        2,
    ));
    manager.finalize_slot(
        "ctx",
        admission.seq_id,
        admission.lease_epoch,
        mirror(&[1, 2, 3]),
        true,
        KvReuseMode::LiveSlotAndSnapshot,
    );
    assert_eq!(manager.pending_prefix_snapshot_count(), 1);

    let warm = manager
        .admit("ctx", KvReuseMode::LiveSlotAndSnapshot, false)
        .expect("warm admission");

    assert_eq!(warm.candidate, CacheCandidate::Live);
    assert_eq!(warm.seq_id, admission.seq_id);
    assert!(warm.lease_epoch > admission.lease_epoch);
    assert_eq!(manager.pending_prefix_snapshot_count(), 0);
}
