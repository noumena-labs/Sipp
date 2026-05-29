//! Terminal output helpers for xtask.

use anyhow::{Context, Result};
use console::{style, Emoji, Term};
use indicatif::{ProgressBar, ProgressStyle};
use std::env;
use std::fmt::Display;
use std::path::Path;
use std::time::{Duration, Instant};
use xshell::Cmd;

const ROCKET: Emoji<'static, 'static> = Emoji("🚀", ">>");
const SPARKLES: Emoji<'static, 'static> = Emoji("✨", "*");
const CHECK: Emoji<'static, 'static> = Emoji("✅", "ok");
const WARN: Emoji<'static, 'static> = Emoji("⚠️", "warning");
const ERROR: Emoji<'static, 'static> = Emoji("❌", "error");
const FOLDER: Emoji<'static, 'static> = Emoji("📁", "path");
const GEAR: Emoji<'static, 'static> = Emoji("⚙️", "*");
const PACKAGE: Emoji<'static, 'static> = Emoji("📦", "artifact");

/// Prints a major build phase header.
pub(crate) fn phase(title: &str) {
    println!();
    println!("{ROCKET} {}", style(title).bold().cyan());
}

/// Prints an informational key-value line.
pub(crate) fn detail(label: &str, value: impl Display) {
    println!(
        "   {SPARKLES} {} {}",
        style(format!("{label}:")).dim(),
        value
    );
}

/// Prints an important filesystem path.
pub(crate) fn path(label: &str, path: &Path) {
    println!(
        "   {FOLDER} {} {}",
        style(format!("{label}:")).dim(),
        style(path.display()).cyan()
    );
}

/// Prints a short step that does not need spinner lifecycle tracking.
pub(crate) fn step(message: impl Display) {
    println!("   {GEAR} {message}");
}

/// Prints a successful status line.
pub(crate) fn success(message: impl Display) {
    println!("   {CHECK} {}", style(message).green());
}

/// Prints an artifact path.
pub(crate) fn artifact(path: &Path) {
    println!(
        "   {PACKAGE} {} {}",
        style("Artifact:").dim(),
        path.display()
    );
}

/// Prints a warning line to stderr.
pub(crate) fn warning(message: impl Display) {
    eprintln!("   {WARN} {}", style(message).yellow());
}

/// Runs an external command with a spinner and elapsed-time reporting.
pub(crate) fn run_command(label: impl Into<String>, command: Cmd<'_>) -> Result<()> {
    let step = TimedStep::start(label);
    let result = command.run();

    match result {
        Ok(()) => {
            step.finish_success();
            Ok(())
        }
        Err(error) => {
            let label = step.label().to_owned();
            step.finish_error();
            Err(error).with_context(|| format!("{label} failed"))
        }
    }
}

/// Formats a list of backends for summaries.
pub(crate) fn backend_list(backends: &[crate::cli::Backend]) -> String {
    if backends.is_empty() {
        return "none".to_owned();
    }

    backends
        .iter()
        .map(crate::cli::Backend::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Formats elapsed time for console output.
pub(crate) fn elapsed(duration: Duration) -> String {
    humantime::format_duration(duration).to_string()
}

struct TimedStep {
    label: String,
    started_at: Instant,
    spinner: Option<ProgressBar>,
}

impl TimedStep {
    fn start(label: impl Into<String>) -> Self {
        let label = label.into();
        let spinner = if use_spinner() {
            let spinner = ProgressBar::new_spinner();
            if let Ok(style) = ProgressStyle::with_template("{spinner:.cyan} {msg}") {
                spinner.set_style(
                    style.tick_strings(&["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]),
                );
            }
            spinner.set_message(label.clone());
            spinner.enable_steady_tick(Duration::from_millis(100));
            Some(spinner)
        } else {
            println!("   {GEAR} {label}");
            None
        };

        Self {
            label,
            started_at: Instant::now(),
            spinner,
        }
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn finish_success(self) {
        let message = format!(
            "{CHECK} {} {}",
            self.label,
            style(format!("({})", elapsed(self.started_at.elapsed()))).dim()
        );

        if let Some(spinner) = self.spinner {
            spinner.finish_with_message(message);
        } else {
            println!("   {message}");
        }
    }

    fn finish_error(self) {
        let message = format!(
            "{ERROR} {} {}",
            self.label,
            style(format!("({})", elapsed(self.started_at.elapsed()))).dim()
        );

        if let Some(spinner) = self.spinner {
            spinner.finish_with_message(message);
        } else {
            eprintln!("   {message}");
        }
    }
}

fn use_spinner() -> bool {
    Term::stderr().is_term() && env::var_os("CI").is_none() && env::var_os("NO_COLOR").is_none()
}
