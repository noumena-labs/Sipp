//! Unit tests for the parent module.

use super::super::*;

#[test]
fn pending_snapshot_dedups_exact_identity_only() {
    let mut cache = PrefixStateCache::default();
    let base = PendingPrefixSnapshot {
        model_fingerprint: 1,
        context_key: "ctx".to_string(),
        seq_id: 0,
        token_count: 2,
        prefix_hash: 10,
        retention_priority: 0,
        prefix_tokens: vec![1, 2],
    };
    cache.enqueue_pending_snapshot(base.clone());
    cache.enqueue_pending_snapshot(PendingPrefixSnapshot {
        retention_priority: 7,
        ..base.clone()
    });
    cache.enqueue_pending_snapshot(PendingPrefixSnapshot {
        token_count: 3,
        prefix_hash: 11,
        prefix_tokens: vec![1, 2, 3],
        ..base
    });

    assert_eq!(cache.pending_snapshots.len(), 2);
}
