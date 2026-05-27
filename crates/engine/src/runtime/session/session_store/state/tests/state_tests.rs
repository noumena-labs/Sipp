//! Unit tests for the parent module.

use super::super::SequenceState;

#[test]
fn sequence_state_reports_sync_with_kv_length() {
    let mut state = SequenceState {
        current_kv_tokens: vec![1, 2, 3],
        n_past: 3,
        ..SequenceState::default()
    };
    assert_eq!(
        usize::try_from(state.n_past),
        Ok(state.current_kv_tokens.len())
    );

    state.n_past = 2;
    assert_ne!(
        usize::try_from(state.n_past),
        Ok(state.current_kv_tokens.len())
    );

    state.n_past = -1;
    assert!(usize::try_from(state.n_past).is_err());
}
