pub(crate) fn normalize_choice(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

pub(crate) fn choice_from_aliases<T: Copy>(value: &str, aliases: &[(&[&str], T)]) -> Option<T> {
    let value = normalize_choice(value);
    aliases
        .iter()
        .find_map(|(choices, choice)| choices.contains(&value.as_str()).then_some(*choice))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestChoice {
        Fast,
        Slow,
    }

    #[test]
    fn choice_alias_lookup_normalizes_input_once() {
        assert_eq!(
            choice_from_aliases(
                " fast-mode ",
                &[
                    (&["fast_mode", "fast"], TestChoice::Fast),
                    (&["slow"], TestChoice::Slow),
                ],
            ),
            Some(TestChoice::Fast)
        );
        assert_eq!(
            choice_from_aliases("missing", &[(&["fast"], TestChoice::Fast)]),
            None
        );
    }
}
