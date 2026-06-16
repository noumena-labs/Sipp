//! Tests the `runtime::inference_runtime::prefill` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use crate::runtime::config::KvReuseMode;
use crate::runtime::metrics::CacheSource;
use crate::runtime::session::CacheCandidate;

use super::prefill::{
    authorized_lcp, live_candidate_lcp, prefix_reuse_plan,
    resolve_initial_decode_context_reservation,
};

#[test]
fn reservation_is_zero_when_no_output_is_requested() {
    assert_eq!(resolve_initial_decode_context_reservation(0, 8), 0);
    assert_eq!(resolve_initial_decode_context_reservation(-1, 8), 0);
}

#[test]
fn reservation_keeps_at_least_one_decode_slot_for_positive_output() {
    assert_eq!(resolve_initial_decode_context_reservation(4, 0), 1);
    assert_eq!(resolve_initial_decode_context_reservation(4, -8), 1);
}

#[test]
fn reservation_is_capped_by_requested_output_tokens() {
    assert_eq!(resolve_initial_decode_context_reservation(2, 8), 2);
    assert_eq!(resolve_initial_decode_context_reservation(8, 2), 2);
}

#[test]
fn prefix_reuse_plan_modes_are_exact() {
    let disabled = prefix_reuse_plan(KvReuseMode::Disabled, false);
    assert!(!disabled.live);
    assert!(!disabled.snapshot);

    let live = prefix_reuse_plan(KvReuseMode::LiveSlotPrefix, false);
    assert!(live.live);
    assert!(!live.snapshot);

    let snapshot = prefix_reuse_plan(KvReuseMode::StateSnapshot, false);
    assert!(!snapshot.live);
    assert!(snapshot.snapshot);

    let both = prefix_reuse_plan(KvReuseMode::LiveSlotAndSnapshot, false);
    assert!(both.live);
    assert!(both.snapshot);
}

#[test]
fn prefix_reuse_plan_bypass_disables_live_and_snapshot_matching() {
    let plan = prefix_reuse_plan(KvReuseMode::LiveSlotAndSnapshot, true);

    assert!(!plan.live);
    assert!(!plan.snapshot);
}

#[test]
fn live_candidate_lcp_requires_explicit_live_candidate() {
    let plan = prefix_reuse_plan(KvReuseMode::LiveSlotPrefix, false);
    let cached = [1, 2, 3];
    let prompt = [1, 2, 4];

    assert_eq!(
        live_candidate_lcp(plan, CacheCandidate::None, &cached, &prompt, true),
        0
    );
    assert_eq!(
        live_candidate_lcp(plan, CacheCandidate::Live, &cached, &prompt, true),
        2
    );

    assert_eq!(
        live_candidate_lcp(plan, CacheCandidate::Live, &[1, 2], &prompt, true),
        2
    );
}

#[test]
fn live_candidate_lcp_allows_generated_suffix_trim_when_supported() {
    let plan = prefix_reuse_plan(KvReuseMode::LiveSlotPrefix, false);
    let final_sequence = [1, 2, 3, 99];
    let repeated_prompt = [1, 2, 3];

    assert_eq!(
        live_candidate_lcp(
            plan,
            CacheCandidate::Live,
            &final_sequence,
            &repeated_prompt,
            true
        ),
        3
    );
}

#[test]
fn live_candidate_lcp_allows_repeated_prompt_when_supported() {
    let plan = prefix_reuse_plan(KvReuseMode::LiveSlotPrefix, false);
    let prompt = [1, 2, 3];

    assert_eq!(
        live_candidate_lcp(plan, CacheCandidate::Live, &prompt, &prompt, true),
        3
    );
}

#[test]
fn live_candidate_lcp_requires_strict_prefix_without_partial_kv() {
    let plan = prefix_reuse_plan(KvReuseMode::LiveSlotPrefix, false);
    let final_sequence = [1, 2, 3, 99];
    let repeated_prompt = [1, 2, 3];

    assert_eq!(
        live_candidate_lcp(
            plan,
            CacheCandidate::Live,
            &final_sequence,
            &repeated_prompt,
            false
        ),
        0
    );
    assert_eq!(
        live_candidate_lcp(
            plan,
            CacheCandidate::Live,
            &repeated_prompt,
            &repeated_prompt,
            false
        ),
        0
    );
    assert_eq!(
        live_candidate_lcp(plan, CacheCandidate::Live, &[1, 2], &repeated_prompt, false),
        2
    );
}

#[test]
fn authorized_lcp_never_rediscovers_reuse_from_none_source() {
    let cached = [1, 2, 3];
    let prompt = [1, 2, 4];

    assert_eq!(authorized_lcp(CacheSource::None, &cached, &prompt), 0);
    assert_eq!(authorized_lcp(CacheSource::Live, &cached, &prompt), 2);
    assert_eq!(authorized_lcp(CacheSource::Snapshot, &cached, &prompt), 2);
}
