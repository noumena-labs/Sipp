//! Animated setup splash for the interactive CogentLM setup flow.
//!
//! This version uses a small character canvas instead of hand-built padding.
//! That keeps the front logo, scan effect, 3D extrusion, and footer aligned
//! in one shared coordinate system.

use anyhow::Result;
use console::{measure_text_width, Term};
use owo_colors::OwoColorize;
use std::thread;
use std::time::Duration;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/terminal/splash_tests.rs"]
mod splash_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const LOGO: &[&str] = &[
    r#"  ____ ___   ____ _____ _   _ _____ _     __  __ "#,
    r#" / ___/ _ \ / ___| ____| \ | |_   _| |   |  \/  |"#,
    r#"| |  | | | | |   |  _| |  \| | | | | |   | |\/| |"#,
    r#"| |__| |_| | |_| | |___| |\  | | | | |___| |  | |"#,
    r#" \____\___/ \____|_____|_| \_| |_| |_____|_|  |_|"#,
];

const TAGLINES: &[&str] = &[
    "LOCAL INFERENCE",
    "WEBGPU RUNTIME",
    "NATIVE BINDINGS",
    "DEVELOPER AUTOMATION",
];

const FRAME_COUNT: u16 = 120;
const FRAME_DELAY: Duration = Duration::from_millis(40);

const FRONT_X: usize = 2;
const FRONT_Y: usize = 1;
const MAX_DEPTH: usize = 6;
const X_SKEW: usize = 2;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Style {
    Plain,
    DimRgb(u8, u8, u8),
    Rgb(u8, u8, u8),
    BoldRgb(u8, u8, u8),
}

#[derive(Clone, Copy)]
struct Cell {
    ch: char,
    style: Style,
    priority: u16,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            ch: ' ',
            style: Style::Plain,
            priority: 0,
        }
    }
}

struct Canvas {
    width: usize,
    height: usize,
    cells: Vec<Cell>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            cells: vec![Cell::default(); width * height],
        }
    }

    fn put(&mut self, x: usize, y: usize, ch: char, style: Style, priority: u16) {
        if ch == ' ' || x >= self.width || y >= self.height {
            return;
        }

        let idx = y * self.width + x;
        if priority >= self.cells[idx].priority {
            self.cells[idx] = Cell {
                ch,
                style,
                priority,
            };
        }
    }

    fn mask(&mut self, x: usize, y: usize, priority: u16) {
        if x >= self.width || y >= self.height {
            return;
        }

        let idx = y * self.width + x;
        if priority >= self.cells[idx].priority {
            self.cells[idx] = Cell {
                ch: ' ',
                style: Style::Plain,
                priority,
            };
        }
    }

    fn draw_text(&mut self, x: usize, y: usize, text: &str, style: Style, priority: u16) {
        for (i, ch) in text.chars().enumerate() {
            self.put(x + i, y, ch, style, priority);
        }
    }

    fn render_lines(&self, global_pad: &str, target_width: usize) -> Vec<String> {
        let mut lines = Vec::with_capacity(self.height);

        for y in 0..self.height {
            let row = &self.cells[y * self.width..(y + 1) * self.width];

            let Some(end) = row.iter().rposition(|cell| cell.ch != ' ') else {
                lines.push(" ".repeat(target_width));
                continue;
            };

            let mut line = String::with_capacity(global_pad.len() + target_width + 64);
            line.push_str(global_pad);

            let mut current_chunk = String::new();
            let mut active_style = Style::Plain;

            for cell in &row[..=end] {
                let effective_style = if cell.ch == ' ' {
                    active_style
                } else {
                    cell.style
                };

                if effective_style != active_style {
                    if !current_chunk.is_empty() {
                        line.push_str(&apply_style(&current_chunk, active_style));
                        current_chunk.clear();
                    }
                    active_style = effective_style;
                }

                current_chunk.push(cell.ch);
            }

            if !current_chunk.is_empty() {
                line.push_str(&apply_style(&current_chunk, active_style));
            }

            let visible_width = measure_text_width(&line);
            if visible_width < target_width {
                line.push_str(&" ".repeat(target_width - visible_width));
            }

            lines.push(line);
        }

        lines
    }
}

fn apply_style(text: &str, style: Style) -> String {
    match style {
        Style::Plain => text.to_string(),
        Style::DimRgb(r, g, b) => text.truecolor(r, g, b).dimmed().to_string(),
        Style::Rgb(r, g, b) => text.truecolor(r, g, b).to_string(),
        Style::BoldRgb(r, g, b) => text.truecolor(r, g, b).bold().to_string(),
    }
}

fn get_terminal_width() -> usize {
    let width = super::inline_width().map(usize::from).unwrap_or_else(|| {
        let (_, width) = Term::stdout().size();
        width as usize
    });

    width.min(80)
}

pub(crate) fn play(no_splash: bool) -> Result<bool> {
    if no_splash || super::no_banner() || !super::inline_active() {
        return Ok(false);
    }

    let term_width = get_terminal_width();

    for frame in 0..FRAME_COUNT {
        if !super::visual_frame(frame_lines(frame, term_width), frame as usize) {
            super::clear_visual();
            return Ok(false);
        }

        thread::sleep(FRAME_DELAY);
    }

    super::clear_visual();
    Ok(true)
}

fn frame_lines(frame: u16, term_width: usize) -> Vec<String> {
    let logo_width = logo_width();
    let logo_height = LOGO.len();

    let min_required_width = FRONT_X + logo_width + 1;
    let max_safe_depth = term_width.saturating_sub(min_required_width) / X_SKEW;
    let dynamic_max_depth = max_safe_depth.min(MAX_DEPTH);

    let depth = animated_depth(frame, dynamic_max_depth);

    let canvas_width = FRONT_X + logo_width + MAX_DEPTH * X_SKEW + 4;
    let canvas_height = FRONT_Y + logo_height + MAX_DEPTH + 5;

    let max_drawn_width = FRONT_X + logo_width + dynamic_max_depth * X_SKEW + 2;
    let global_pad = " ".repeat(term_width.saturating_sub(max_drawn_width) / 2);

    let mut canvas = Canvas::new(canvas_width, canvas_height);

    draw_back_glow(&mut canvas, frame, logo_width, logo_height, depth);
    draw_extrusion(&mut canvas, frame, logo_width, depth);

    for (row, line) in LOGO.iter().enumerate() {
        let padded = pad_to_width(line, logo_width);

        if let Some(first) = padded.chars().position(|c| c != ' ') {
            let last = padded.trim_end().chars().count().saturating_sub(1);

            for col in first..=last {
                canvas.mask(FRONT_X + col, FRONT_Y + row, 80);
            }
        }
    }

    draw_front_face(&mut canvas, frame, logo_width);
    draw_bottom_ribs(&mut canvas, frame, logo_width, logo_height, depth);
    draw_footer(&mut canvas, frame, logo_width);

    canvas.render_lines(&global_pad, term_width)
}

fn draw_back_glow(
    canvas: &mut Canvas,
    frame: u16,
    logo_width: usize,
    logo_height: usize,
    depth: usize,
) {
    let glow_width = logo_width / 2;
    let y = FRONT_Y + logo_height + depth;
    let pulse = ((frame as usize / 3) % 6).min(5);

    if y >= canvas.height || glow_width < 8 {
        return;
    }

    let start_x = FRONT_X + logo_width / 4 + pulse;
    let style = Style::DimRgb(75, 0, 130);

    canvas.put(start_x, y, '◢', style, 5);
    for i in 0..glow_width {
        canvas.put(start_x + 1 + i, y, '▀', style, 5);
    }
    canvas.put(start_x + glow_width + 1, y, '◣', style, 5);
}

fn draw_extrusion(canvas: &mut Canvas, frame: u16, logo_width: usize, depth: usize) {
    let mask = logo_mask(logo_width);

    for layer in (1..=depth).rev() {
        let x_offset = layer * X_SKEW;
        let y_offset = layer;

        let r = 40u8.saturating_add(layer as u8 * 20);
        let g = 10u8.saturating_add(layer as u8 * 4);
        let b = 110u8.saturating_add(layer as u8 * 16);
        let base_style = Style::DimRgb(r, g, b);

        for (row, line) in LOGO.iter().enumerate() {
            let padded = pad_to_width(line, logo_width);

            for (col, ch) in padded.chars().enumerate() {
                if ch == ' ' {
                    continue;
                }
                let Some(edge) = extrusion_edge(&mask, row, col) else {
                    continue;
                };

                let style = if is_extrusion_glint(frame, row, col, layer) {
                    Style::BoldRgb(0, 240, 255)
                } else {
                    base_style
                };

                // FIX: Loop across the full X_SKEW horizontal width to fill the empty column
                // gaps and seamlessly tie the extrusion chunks together.
                for s in 0..X_SKEW {
                    let glyph = match edge {
                        ExtrusionEdge::Side if s > 0 => '█',   // Solid block filler
                        ExtrusionEdge::Corner if s > 0 => '█', // Solid block filler
                        _ => edge.glyph(),
                    };

                    canvas.put(
                        FRONT_X + x_offset + col - s,
                        FRONT_Y + y_offset + row,
                        glyph,
                        style,
                        30 - layer as u16,
                    );
                }
            }
        }
    }
}

fn draw_front_face(canvas: &mut Canvas, frame: u16, logo_width: usize) {
    let scan = scan_column(frame, logo_width);

    for (row, line) in LOGO.iter().enumerate() {
        let padded = pad_to_width(line, logo_width);

        for (col, ch) in padded.chars().enumerate() {
            if ch == ' ' {
                continue;
            }

            let distance = col.abs_diff(scan);
            let style = front_scan_style(frame, row, distance);

            canvas.put(FRONT_X + col, FRONT_Y + row, ch, style, 100);
        }
    }
}

fn draw_bottom_ribs(
    canvas: &mut Canvas,
    frame: u16,
    logo_width: usize,
    logo_height: usize,
    depth: usize,
) {
    let base_y = FRONT_Y + logo_height;

    for rib in 0..=depth {
        let y = base_y + rib;
        if y >= canvas.height.saturating_sub(3) {
            break;
        }

        let start_x = FRONT_X + logo_width / 5 + rib * X_SKEW / 2;
        let width = logo_width
            .saturating_sub(logo_width / 5)
            .saturating_add(depth * X_SKEW)
            .saturating_sub(rib * 2);

        if width < 8 || start_x + width + 2 >= canvas.width {
            continue;
        }

        let shimmer = ((frame as usize + rib * 4) % 15) < 3;
        let shade = 60u8.saturating_sub(rib as u8 * 5);

        let style = if shimmer {
            Style::Rgb(0, 220, 255)
        } else {
            Style::DimRgb(shade + 10, 20, shade + 50)
        };

        canvas.put(start_x, y, '▕', style, 20);
        for i in 0..width {
            let ch = if i % 5 == 0 { '┼' } else { '─' };
            canvas.put(start_x + 1 + i, y, ch, style, 20);
        }
        canvas.put(start_x + width + 1, y, '▏', style, 20);
    }
}

fn draw_footer(canvas: &mut Canvas, frame: u16, logo_width: usize) {
    let boot_y = canvas.height.saturating_sub(2);
    let tag_y = canvas.height.saturating_sub(1);
    let logo_center = FRONT_X + (logo_width / 2);

    let boot_full = format!("▲ INITIALIZING CORE ENGINE {} ", spinner(frame));
    let boot_text = typewriter(&boot_full, frame);
    let boot_x = logo_center.saturating_sub(char_len(&boot_full) / 2);

    canvas.draw_text(boot_x, boot_y, &boot_text, Style::Rgb(0, 255, 170), 200);

    let tagline = TAGLINES[((frame as usize) / 16) % TAGLINES.len()];
    let prefix = "◢◤ COGENT // ";
    let suffix = " ◥◣";
    let total = char_len(prefix) + char_len(tagline) + char_len(suffix);
    let x = logo_center.saturating_sub(total / 2);

    canvas.draw_text(x, tag_y, prefix, Style::Rgb(255, 0, 128), 210);
    canvas.draw_text(
        x + char_len(prefix),
        tag_y,
        tagline,
        Style::BoldRgb(255, 230, 0),
        211,
    );
    canvas.draw_text(
        x + char_len(prefix) + char_len(tagline),
        tag_y,
        suffix,
        Style::Rgb(255, 0, 128),
        210,
    );
}

fn front_color(frame: u16, row: usize) -> Style {
    let time = (frame as f32) * 0.12 + (row as f32) * 0.35;
    let r = ((time.sin() * 25.0) + 230.0) as u8;
    let g = (((time + 1.5).cos() * 45.0) + 55.0) as u8;
    let b = 255;

    Style::Rgb(r, g, b)
}

fn spinner(frame: u16) -> &'static str {
    const SPINNER: &[&str] = &["▰▱▱▱▱", "▰▰▱▱▱", "▰▰▰▱▱", "▰▰▰▰▱", "▰▰▰▰▰", "▱▱▱▱▱"];
    SPINNER[(frame as usize / 2) % SPINNER.len()]
}

fn typewriter(full: &str, frame: u16) -> String {
    let full_len = char_len(full);
    let typed_len = ((frame as usize) / 2).min(full_len);

    let typed: String = full.chars().take(typed_len).collect();

    let cursor = if frame % 4 < 2 && typed_len < full_len {
        "█"
    } else {
        " "
    };

    format!("{typed}{cursor}")
}

fn scan_column(frame: u16, width: usize) -> usize {
    if width <= 1 {
        return 0;
    }

    let cycle = (width - 1) * 2;
    let progress = ((frame as usize) * 2) % cycle;

    if progress < width {
        progress
    } else {
        cycle - progress
    }
}

fn animated_depth(frame: u16, dynamic_max: usize) -> usize {
    let intro = (frame as usize / 3).min(dynamic_max);
    let pulse = if frame > 18 {
        (frame as usize / 6) % 2
    } else {
        0
    };

    (intro + pulse).min(dynamic_max)
}

fn front_scan_style(frame: u16, row: usize, distance: usize) -> Style {
    match distance {
        0 => Style::BoldRgb(255, 255, 255),
        1 => Style::BoldRgb(0, 245, 255),
        2 => Style::Rgb(210, 0, 255),
        3 => Style::DimRgb(120, 0, 180),
        _ => front_color(frame, row),
    }
}

fn is_extrusion_glint(frame: u16, row: usize, col: usize, layer: usize) -> bool {
    (frame as usize + row * 3 + col + layer * 5) % 67 == 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExtrusionEdge {
    Bottom,
    Side,
    Corner,
}

impl ExtrusionEdge {
    fn glyph(self) -> char {
        match self {
            Self::Bottom => '▄',
            Self::Side => '▐',
            Self::Corner => '▞',
        }
    }
}

fn logo_mask(width: usize) -> Vec<Vec<bool>> {
    LOGO.iter()
        .map(|line| {
            pad_to_width(line, width)
                .chars()
                .map(|ch| ch != ' ')
                .collect()
        })
        .collect()
}

fn extrusion_edge(mask: &[Vec<bool>], row: usize, col: usize) -> Option<ExtrusionEdge> {
    if !mask_cell(mask, row, col) {
        return None;
    }

    let open_right = !mask_cell(mask, row, col + 1);
    let open_below = !mask_cell(mask, row + 1, col);

    match (open_right, open_below) {
        (true, true) => Some(ExtrusionEdge::Corner),
        (true, false) => Some(ExtrusionEdge::Side),
        (false, true) => Some(ExtrusionEdge::Bottom),
        (false, false) => None,
    }
}

fn mask_cell(mask: &[Vec<bool>], row: usize, col: usize) -> bool {
    mask.get(row)
        .and_then(|line| line.get(col))
        .copied()
        .unwrap_or(false)
}

fn logo_width() -> usize {
    LOGO.iter().map(|line| char_len(line)).max().unwrap_or(0)
}

fn char_len(text: &str) -> usize {
    text.chars().count()
}

fn pad_to_width(text: &str, width: usize) -> String {
    let len = char_len(text);

    if len >= width {
        text.to_owned()
    } else {
        let mut out = String::with_capacity(text.len() + width - len);
        out.push_str(text);
        out.push_str(&" ".repeat(width - len));
        out
    }
}
