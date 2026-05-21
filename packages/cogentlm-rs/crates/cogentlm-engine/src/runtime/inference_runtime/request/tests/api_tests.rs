use crate::runtime::inference_runtime::request::api::normalize_stop_sequences;

#[test]
fn normalize_stop_sequences_drops_empty_and_deduplicates() {
    let normalized = normalize_stop_sequences(vec![
        "zz".to_string(),
        String::new(),
        "aa".to_string(),
        "zz".to_string(),
    ]);

    assert_eq!(normalized, ["aa", "zz"]);
}
