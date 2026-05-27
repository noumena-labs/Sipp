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
    mod collection_tests;
}
