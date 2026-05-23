use std::collections::BTreeMap;

pub(crate) fn sorted_values<T: Ord>(mut values: Vec<T>) -> Vec<T> {
    values.sort();
    values
}

pub(crate) fn sorted_copied_values<T: Copy + Ord>(values: impl IntoIterator<Item = T>) -> Vec<T> {
    let mut values: Vec<_> = values.into_iter().collect();
    values.sort_unstable();
    values
}

pub(crate) fn sorted_unique_strings(mut values: Vec<String>) -> Vec<String> {
    values = sorted_values(values);
    values.dedup();
    values
}

pub(crate) fn sorted_unique_non_empty_strings(
    values: impl IntoIterator<Item = String>,
) -> Vec<String> {
    sorted_unique_strings(
        values
            .into_iter()
            .filter(|value| !value.is_empty())
            .collect(),
    )
}

pub(crate) fn sorted_unique_strings_with_optional(
    mut values: Vec<String>,
    optional: Option<&String>,
) -> Vec<String> {
    if let Some(value) = optional {
        values.push(value.clone());
    }
    sorted_unique_strings(values)
}

pub(crate) fn sorted_ref_deltas(
    previous_refs: &[String],
    updated_refs: &[String],
) -> (Vec<String>, Vec<String>) {
    let mut removed_refs = Vec::with_capacity(previous_refs.len());
    let mut added_refs = Vec::with_capacity(updated_refs.len());
    let mut previous = previous_refs.iter();
    let mut updated = updated_refs.iter();
    let mut previous_id = previous.next();
    let mut updated_id = updated.next();

    loop {
        match (previous_id, updated_id) {
            (Some(previous_value), Some(updated_value)) => {
                match previous_value.cmp(updated_value) {
                    std::cmp::Ordering::Less => {
                        removed_refs.push(previous_value.clone());
                        previous_id = previous.next();
                    }
                    std::cmp::Ordering::Equal => {
                        previous_id = previous.next();
                        updated_id = updated.next();
                    }
                    std::cmp::Ordering::Greater => {
                        added_refs.push(updated_value.clone());
                        updated_id = updated.next();
                    }
                }
            }
            (Some(previous_value), None) => {
                removed_refs.push(previous_value.clone());
                removed_refs.extend(previous.cloned());
                break;
            }
            (None, Some(updated_value)) => {
                added_refs.push(updated_value.clone());
                added_refs.extend(updated.cloned());
                break;
            }
            (None, None) => break,
        }
    }

    (removed_refs, added_refs)
}

pub(crate) fn remove_matching_values<T>(
    values: &mut BTreeMap<String, T>,
    mut predicate: impl FnMut(&T) -> bool,
) -> Vec<T> {
    let removed_ids: Vec<_> = values
        .iter()
        .filter_map(|(id, value)| predicate(value).then_some(id.clone()))
        .collect();
    let mut removed_values = Vec::with_capacity(removed_ids.len());
    for id in removed_ids {
        if let Some(value) = values.remove(&id) {
            removed_values.push(value);
        }
    }
    removed_values
}

#[cfg(test)]
mod tests {
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
}
