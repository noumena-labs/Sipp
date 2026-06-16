//! Tests the `runtime::session::prefix_state_cache::state_io` module in
//! `sipp`.
//!
//! Covers snapshot state store/restore validation with in-memory state bytes
//! and empty native handles; native capture is intentionally skipped.

use crate::native_bridge::NativeRuntimeHandle;

use super::super::{PendingPrefixSnapshot, PrefixStateStoreRequest};
use super::*;

fn request<'a>(
    state_bytes: Vec<u8>,
    seq_id: i32,
    tokens: &'a [i32],
    token_count: usize,
) -> PrefixStateStoreRequest<'a> {
    PrefixStateStoreRequest {
        state_bytes,
        seq_id,
        model_fingerprint: 7,
        snapshot_scope: "ctx",
        tokens,
        token_count,
        prefix_hash: 99,
        retention_priority: 3,
    }
}

fn pending_snapshot(
    seq_id: i32,
    generation: u64,
    snapshot_scope: &str,
    tokens: Vec<i32>,
) -> PendingPrefixSnapshot {
    PendingPrefixSnapshot {
        seq_id,
        generation,
        model_fingerprint: 7,
        snapshot_scope: snapshot_scope.to_string(),
        token_count: tokens.len(),
        prefix_hash: 99,
        retention_priority: 3,
        prefix_tokens: tokens,
    }
}

#[test]
fn store_prefix_state_rejects_invalid_requests_without_mutation() {
    let tokens = [1, 2, 3];
    let mut cache = PrefixStateCache::new(4, 1024);

    assert!(!cache.store_prefix_state(request(vec![1], -1, &tokens, 1)));
    assert!(!cache.store_prefix_state(request(vec![1], 0, &tokens, 0)));
    assert!(!cache.store_prefix_state(request(vec![1], 0, &tokens, 4)));
    assert!(!cache.store_prefix_state(request(Vec::new(), 0, &tokens, 1)));
    assert!(cache.entries.is_empty());
    assert_eq!(cache.total_approx_bytes, 0);
}

#[test]
fn store_prefix_state_inserts_prefix_tokens_and_state_bytes() {
    let tokens = [1, 2, 3];
    let mut cache = PrefixStateCache::new(4, 1024);

    assert!(cache.store_prefix_state(request(vec![9, 8], 0, &tokens, 2)));

    assert_eq!(cache.entries.len(), 1);
    let entry = &cache.entries[0];
    assert_eq!(entry.model_fingerprint, 7);
    assert_eq!(entry.snapshot_scope, "ctx");
    assert_eq!(entry.prefix_tokens, [1, 2]);
    assert_eq!(entry.state_bytes, [9, 8]);
    assert_eq!(entry.token_count, 2);
    assert_eq!(entry.prefix_hash, 99);
    assert!(entry.approx_bytes >= entry.state_bytes.len());
}

#[test]
fn capture_prefix_state_rejects_empty_runtime_without_storing_entry() {
    let tokens = [1, 2, 3];
    let mut cache = PrefixStateCache::new(4, 1024);
    let runtime = NativeRuntimeHandle::empty_for_tests();

    assert!(!cache.capture_prefix_state(&runtime, 0, 7, "ctx", &tokens, 2, 99, 3));
    assert!(cache.entries.is_empty());
    assert_eq!(cache.total_approx_bytes, 0);
}

#[test]
fn restore_prefix_state_rejects_negative_seq_or_empty_runtime() {
    let tokens = [1, 2, 3];
    let mut cache = PrefixStateCache::new(4, 1024);
    assert!(cache.store_prefix_state(request(vec![9, 8], 0, &tokens, 2)));
    let entry = cache.entries[0].clone();
    let mut runtime = NativeRuntimeHandle::empty_for_tests();

    assert!(!cache.restore_prefix_state(&mut runtime, -1, &entry));
    assert!(!cache.restore_prefix_state(&mut runtime, 0, &entry));
}

#[test]
fn enqueue_pending_snapshot_coalesces_exact_identity() {
    let mut cache = PrefixStateCache::new(4, 1024);
    let mut replacement = pending_snapshot(0, 11, "ctx", vec![1, 2]);
    replacement.retention_priority = 99;

    cache.enqueue_pending_snapshot(pending_snapshot(0, 11, "ctx", vec![1, 2]));
    cache.enqueue_pending_snapshot(replacement);

    assert_eq!(cache.pending_snapshot_count(), 1);
    assert_eq!(cache.pending_snapshots[0].retention_priority, 99);
}

#[test]
fn drop_pending_snapshots_for_seq_removes_only_matching_sequence() {
    let mut cache = PrefixStateCache::new(4, 1024);
    cache.enqueue_pending_snapshot(pending_snapshot(0, 11, "a", vec![1, 2]));
    cache.enqueue_pending_snapshot(pending_snapshot(1, 11, "b", vec![3, 4]));

    cache.drop_pending_snapshots_for_seq(0);

    assert_eq!(cache.pending_snapshot_count(), 1);
    assert_eq!(cache.pending_snapshots[0].seq_id, 1);
}

#[test]
fn drain_pending_snapshots_drops_stale_generation_without_store() {
    let mut cache = PrefixStateCache::new(4, 1024);
    let runtime = NativeRuntimeHandle::empty_for_tests();
    cache.enqueue_pending_snapshot(pending_snapshot(0, 11, "ctx", vec![1, 2]));

    let drained =
        cache.drain_pending_snapshots(&runtime, 2, |_seq_id, generation| generation == 12, |_| {});

    assert_eq!(drained, 1);
    assert_eq!(cache.pending_snapshot_count(), 0);
    assert!(cache.entries.is_empty());
}
