//! Tests the `runtime::session::prefix_state_cache::state_io` module in
//! `cogentlm-engine`.
//!
//! Covers snapshot state store/restore validation with in-memory state bytes
//! and empty native handles; native capture is intentionally skipped.

use crate::native_bridge::NativeRuntimeHandle;

use super::super::PrefixStateStoreRequest;
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
