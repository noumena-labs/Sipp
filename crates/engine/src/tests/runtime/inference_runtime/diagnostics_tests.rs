use super::diagnostics::NoProgressCounts;

#[test]
fn no_progress_diagnostic_includes_all_counters() {
    let message = NoProgressCounts {
        active: 1,
        decode_ready: 2,
        prefill_ready: 3,
        decode_without_seed: 4,
        emit_without_buffer: 5,
    }
    .to_message();

    assert!(message.contains("active=1"));
    assert!(message.contains("decode_ready=2"));
    assert!(message.contains("prefill_ready=3"));
    assert!(message.contains("decode_without_seed=4"));
    assert!(message.contains("emit_without_buffer=5"));
}
