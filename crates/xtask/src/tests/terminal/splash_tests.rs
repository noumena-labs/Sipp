//! Unit tests for the parent module.

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
