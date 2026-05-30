//! Animated setup splash for the interactive CogentLM setup flow.

use crate::output;
use anyhow::Result;
use std::thread;
use std::time::Duration;

const LOGO: &[&str] = &[
    r#"  ____ ___   ____ _____ _   _ _____ _     __  __ "#,
    r#" / ___/ _ \ / ___| ____| \ | |_   _| |   |  \/  |"#,
    r#"| |  | | | | |  _|  _| |  \| | | | | |   | |\/| |"#,
    r#"| |__| |_| | |_| | |___| |\  | | | | |___| |  | |"#,
    r#" \____\___/ \____|_____|_| \_| |_| |_____|_|  |_|"#,
];
const TAGLINES: &[&str] = &[
    "local inference",
    "webgpu runtime",
    "native bindings",
    "developer automation",
];
const FRAME_COUNT: u16 = 34;
const FRAME_DELAY: Duration = Duration::from_millis(70);
const MAX_DEPTH: u16 = 5;

/// Plays the setup splash inside the bounded inline viewport when available.
pub(crate) fn play(no_splash: bool) -> Result<bool> {
    if no_splash || output::no_banner() || !output::inline_active() {
        return Ok(false);
    }

    for frame in 0..FRAME_COUNT {
        if !output::visual_frame(frame_lines(frame), frame as usize) {
            return Ok(false);
        }
        thread::sleep(FRAME_DELAY);
    }

    output::clear_visual();
    Ok(true)
}

fn frame_lines(frame: u16) -> Vec<String> {
    let depth = animated_depth(frame);
    let tagline = TAGLINES[((frame / 8) as usize) % TAGLINES.len()];
    let logo_width = logo_width();
    let scan_column = ((frame * 3) as usize) % logo_width;
    let mut lines = Vec::new();

    lines.push(format!(
        "{}{}",
        " ".repeat(depth as usize),
        format!("/{}\\", "_".repeat(logo_width.saturating_sub(2)))
    ));

    for (row, logo_line) in LOGO.iter().enumerate() {
        let mut line = logo_line.to_string();
        if scan_column < line.len() {
            line.replace_range(scan_column..scan_column + 1, "|");
        }

        let side = if (frame + row as u16) % 2 == 0 { "/" } else { "\\" };
        lines.push(format!(
            "{}{} {}",
            " ".repeat((MAX_DEPTH - depth) as usize + row % 2),
            line,
            side
        ));
    }

    for layer in 0..depth {
        lines.push(format!(
            "{}\\{}\\",
            " ".repeat(layer as usize + 2),
            "_".repeat(logo_width.saturating_sub(layer as usize + 4))
        ));
    }

    lines.push("".to_owned());
    lines.push("bootstrapping local development".to_owned());
    lines.push(format!(":: COGENTLM {tagline} ::"));
    lines
}

fn animated_depth(frame: u16) -> u16 {
    let grow = frame.min(MAX_DEPTH);
    let pulse = if frame > 20 { (frame - 20) % 3 } else { 0 };
    (grow + pulse).min(MAX_DEPTH)
}

fn logo_width() -> usize {
    LOGO.iter().map(|line| line.len()).max().unwrap_or(0)
}
