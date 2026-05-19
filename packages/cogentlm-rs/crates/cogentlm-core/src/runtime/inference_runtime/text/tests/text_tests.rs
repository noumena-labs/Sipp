//! Unit tests for the parent module.

use super::super::*;

#[test]
fn detects_incomplete_utf8_tail() {
    assert_eq!(incomplete_utf8_tail_length(b"a"), 0);
    assert_eq!(incomplete_utf8_tail_length(&[0xE2, 0x82]), 2);
    assert_eq!(incomplete_utf8_tail_length(&[0xE2, 0x82, 0xAC]), 0);
}

#[test]
fn earliest_stop_index_split_finds_cross_boundary_matches() {
    let stops = vec!["🌍!".to_string(), "stop".to_string(), "\n".to_string()];

    assert_eq!(
        earliest_stop_index_split("hello st", "op world", &stops),
        Some(6)
    );
    assert_eq!(
        earliest_stop_index_split("hello, 🌍", "! world", &stops),
        Some(7)
    );
    assert_eq!(
        earliest_stop_index_split("hello", "world stop", &stops),
        Some(11)
    );
    assert_eq!(earliest_stop_index_split("hello", "world", &stops), None);
}
