//! Tests the `terminal` module in `xtask`.
//!
//! Covers ANSI parsing, truncation, command-log labeling, subprocess output
//! tailing, and progress classification with synthetic output instead of
//! running external commands.

use console::measure_text_width;
use crossterm::style::Color;
use std::process::{ExitStatus, Output};

use super::*;

#[cfg(unix)]
fn success_status() -> ExitStatus {
    use std::os::unix::process::ExitStatusExt;

    ExitStatus::from_raw(0)
}

#[cfg(windows)]
fn success_status() -> ExitStatus {
    use std::os::windows::process::ExitStatusExt;

    ExitStatus::from_raw(0)
}

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

#[test]
fn ansi_parser_handles_indexed_colors_resets_tabs_and_osc_sequences() {
    let segments =
        parse_ansi_segments("\u{1b}]0;title\u{7}\u{1b}[38;5;42;2mhi\t\u{1b}[22;39mthere");

    assert_eq!(segments.len(), 2);
    assert_eq!(segments[0].text, "hi ");
    assert_eq!(segments[0].color, Color::AnsiValue(42));
    assert!(segments[0].dim);
    assert_eq!(segments[1].text, "there");
    assert_eq!(segments[1].color, Color::White);
    assert!(!segments[1].bold);
    assert!(!segments[1].dim);
}

#[test]
fn segment_truncation_preserves_style_and_handles_tiny_widths() {
    let segments = vec![RowSegment::styled("abcdef", Color::Blue, true, false)];

    assert!(truncate_segments(&segments, 0).is_empty());
    assert_eq!(truncate_segments(&segments, 2)[0].text, "..");
    let truncated = truncate_segments(&segments, 5);
    assert_eq!(measure_text_width(&truncated[0].text), 5);
    assert_eq!(truncated[0].color, Color::Blue);
    assert!(truncated[0].bold);
}

#[test]
fn command_log_labels_are_sanitized_and_have_fallbacks() {
    assert_eq!(sanitize_log_label("Build Node.js!"), "build-node-js");
    assert_eq!(sanitize_log_label("!!!"), "command");
    assert!(command_log_file_name("Build Node").ends_with("-build-node.log"));
}

#[test]
fn tail_output_lines_combines_stdout_and_stderr_without_blank_lines() {
    let output = Output {
        status: success_status(),
        stdout: b"one\n\n two \nthree\n".to_vec(),
        stderr: b"warning\nerror\n".to_vec(),
    };

    assert_eq!(
        tail_output_lines(&output, 3),
        vec!["three".to_owned(), "warning".to_owned(), "error".to_owned()]
    );
    assert_eq!(command_status_label(&output), "exit code 0");
}

#[test]
fn output_stream_classification_detects_errors_warnings_success_and_progress() {
    let (label, kind) = OutputStream::Stdout.classify("error: failed");
    assert_eq!(label, "ERR");
    assert_eq!(kind.color(), Color::Red);

    let (label, kind) = OutputStream::Stderr.classify("warning: heads up");
    assert_eq!(label, "WARN");
    assert_eq!(kind.color(), Color::Yellow);

    let (label, kind) = OutputStream::Pty.classify("Finished release");
    assert_eq!(label, "OK");
    assert_eq!(kind.color(), Color::Green);

    let (label, kind) = OutputStream::Stdout.classify("Compiling xtask");
    assert_eq!(label, "BUILD");
    assert_eq!(kind.color(), Color::Blue);
    assert!(is_build_progress_line("installing dependencies"));
}

#[test]
fn inline_and_visual_helpers_are_deterministic() {
    assert!(!should_use_inline(true, false));
    assert!(!should_use_inline(false, true));
    assert_eq!(spinner_frame(0), ".  ");
    assert_eq!(spinner_frame(5), "   ");
    assert_eq!(visual_line_segments("plain", 0)[0].text, "plain");
}
