//! Tests the `terminal` module in `xtask`.
//!
//! Covers developer automation helpers, catalog logic, and terminal formatting with deterministic fixtures instead of invoking external toolchains.

use console::measure_text_width;
use crossterm::style::Color;

use super::*;

#[test]
fn ansi_truecolor_segments_preserve_rgb_style() {
    let segments = parse_ansi_segments("\u{1b}[38;2;12;34;56;1mhi\u{1b}[0m");

    assert_eq!(segments.len(), 1);
    assert_eq!(segments[0].text, "hi");
    assert_eq!(
        segments[0].color,
        Color::Rgb {
            r: 12,
            g: 34,
            b: 56
        }
    );
    assert!(segments[0].bold);
}

#[test]
fn truncate_text_uses_visible_ansi_width() {
    let text = "\u{1b}[31mabcdef\u{1b}[0m";
    let truncated = truncate_text(text, 4);

    assert_eq!(measure_text_width(&truncated), 4);
}
