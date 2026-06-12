//! Tests the `collection` module in `cogentlm`.
//!
//! Covers engine public values and helper behavior with deterministic unit fixtures; model-backed checks stay explicitly ignored.

use super::*;

#[test]
fn sorted_ref_deltas_reports_linear_adds_and_removals() {
    let previous = vec![
        "asset-a".to_string(),
        "asset-c".to_string(),
        "asset-e".to_string(),
    ];
    let updated = vec![
        "asset-b".to_string(),
        "asset-c".to_string(),
        "asset-f".to_string(),
    ];

    let (removed, added) = sorted_ref_deltas(&previous, &updated);

    assert_eq!(removed, vec!["asset-a", "asset-e"]);
    assert_eq!(added, vec!["asset-b", "asset-f"]);
}

#[test]
fn sorted_ref_deltas_reports_no_changes_for_equal_refs() {
    let refs = vec!["asset-a".to_string(), "asset-b".to_string()];

    let (removed, added) = sorted_ref_deltas(&refs, &refs);

    assert!(removed.is_empty());
    assert!(added.is_empty());
}

#[test]
fn sorted_helpers_preserve_duplicates() {
    let strings = vec!["b".to_string(), "a".to_string(), "a".to_string()];

    assert_eq!(sorted_values(strings), vec!["a", "a", "b"]);
    assert_eq!(sorted_copied_values([3, 1, 2, 1]), vec![1, 1, 2, 3]);
}

#[test]
fn remove_matching_values_preserves_sorted_key_order() {
    let mut values = BTreeMap::from([
        ("b".to_string(), 2),
        ("a".to_string(), 1),
        ("c".to_string(), 3),
    ]);

    let removed = remove_matching_values(&mut values, |value| value % 2 == 1);

    assert_eq!(removed, vec![1, 3]);
    assert_eq!(values, BTreeMap::from([("b".to_string(), 2)]));
}
