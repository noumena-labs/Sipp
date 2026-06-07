//! Tests the `terminal::splash` module in `xtask`.
//!
//! Covers frame sizing, mask edges, scan/depth animation helpers, and string
//! padding/typewriter boundaries without playing the timed terminal animation.

use console::measure_text_width;

use super::*;

#[test]
fn frame_lines_keep_a_fixed_visible_width() {
    let width = 74;

    for frame in [0, 18, 48, FRAME_COUNT - 1] {
        let lines = frame_lines(frame, width);

        assert!(!lines.is_empty());
        assert!(lines.iter().all(|line| measure_text_width(line) == width));
    }
}

#[test]
fn frame_lines_do_not_add_glitch_characters() {
    let lines = frame_lines(18, 74);

    assert!(lines.iter().all(|line| !line.contains('*')));
    assert!(lines.iter().all(|line| !line.contains('+')));
}

#[test]
fn extrusion_uses_only_silhouette_edges() {
    let mask = vec![vec![true, true], vec![true, true]];

    assert_eq!(extrusion_edge(&mask, 0, 0), None);
    assert_eq!(extrusion_edge(&mask, 0, 1), Some(ExtrusionEdge::Side));
    assert_eq!(extrusion_edge(&mask, 1, 0), Some(ExtrusionEdge::Bottom));
    assert_eq!(extrusion_edge(&mask, 1, 1), Some(ExtrusionEdge::Corner));
}

#[test]
fn scan_column_bounces_across_logo_width() {
    assert_eq!(scan_column(0, 0), 0);
    assert_eq!(scan_column(0, 1), 0);
    assert_eq!(scan_column(0, 4), 0);
    assert_eq!(scan_column(1, 4), 2);
    assert_eq!(scan_column(2, 4), 2);
}

#[test]
fn animated_depth_is_bounded_by_dynamic_max() {
    assert_eq!(animated_depth(0, 4), 0);
    assert!(animated_depth(18, 4) <= 4);
    assert_eq!(animated_depth(120, 2), 2);
}

#[test]
fn typewriter_and_padding_helpers_respect_character_width() {
    let typed = typewriter("abcdef", 4);
    assert!(typed.starts_with("ab"));
    assert!(char_len(&typed) <= char_len("abcdef") + 1);
    assert_eq!(pad_to_width("abc", 5), "abc  ");
    assert_eq!(pad_to_width("abcdef", 3), "abcdef");
}

#[test]
fn mask_and_logo_helpers_handle_out_of_bounds_cells() {
    let mask = vec![vec![true]];

    assert!(mask_cell(&mask, 0, 0));
    assert!(!mask_cell(&mask, 1, 0));
    assert!(logo_width() > 0);
    assert_eq!(spinner(0), spinner(12));
    assert!(matches!(
        front_scan_style(0, 0, 0),
        Style::BoldRgb(255, 255, 255)
    ));
}
