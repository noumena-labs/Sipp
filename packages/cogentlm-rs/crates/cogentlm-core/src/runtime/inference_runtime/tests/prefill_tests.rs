use super::super::prefill::resolve_initial_decode_context_reservation;

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
