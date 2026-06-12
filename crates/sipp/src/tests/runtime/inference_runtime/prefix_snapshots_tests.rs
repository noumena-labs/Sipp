//! Tests the `runtime::inference_runtime::prefix_snapshots` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::decode_seed_snapshot_token_count;

#[test]
fn decode_seed_snapshot_requires_at_least_two_prompt_tokens() {
    assert_eq!(decode_seed_snapshot_token_count(0), None);
    assert_eq!(decode_seed_snapshot_token_count(1), None);
    assert_eq!(decode_seed_snapshot_token_count(2), Some(1));
    assert_eq!(decode_seed_snapshot_token_count(19), Some(18));
}
