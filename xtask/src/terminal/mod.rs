//! Terminal output helpers for xtask.

pub(crate) mod splash;

use crate::utils::BuildContext;
use anyhow::{anyhow, Context, Result};
use console::{measure_text_width, style, truncate_str, Term};
use crossterm::cursor::{position, Hide, MoveTo, MoveToPreviousLine, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType, DisableLineWrap, EnableLineWrap};
use crossterm::{execute, queue};
use indicatif::{ProgressBar, ProgressStyle};
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Read, Write as _};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tui_banner::{Align, Banner, Style as BannerStyle};
use xshell::Cmd;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/terminal_tests.rs"]
mod terminal_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const INLINE_DEFAULT_LINES: u16 = 18;
const INLINE_MIN_LINES: u16 = 12;
const INLINE_MIN_COLUMNS: u16 = 60;
const INLINE_MIN_RENDER_INTERVAL: Duration = Duration::from_millis(50);
const MAX_ACTIVITY_LINES: usize = 160;
const MAX_OUTPUT_LINES: usize = 400;

static STREAM_SUBPROCESS: AtomicBool = AtomicBool::new(false);
static NO_BANNER: AtomicBool = AtomicBool::new(false);
static COMMAND_LOG_COUNTER: AtomicUsize = AtomicUsize::new(0);
static COMMAND_LOG_DIR: OnceLock<PathBuf> = OnceLock::new();
static WORKSPACE_ROOT: OnceLock<PathBuf> = OnceLock::new();
static INLINE_RENDERER: OnceLock<Mutex<Option<InlineRenderer>>> = OnceLock::new();

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommandDisplay {
    General,
    Build,
    Test,
}

impl CommandDisplay {
    fn shows_output_line(self, label: &str, normalized: &str) -> bool {
        match self {
            CommandDisplay::General => true,
            CommandDisplay::Build => {
                matches!(label, "BUILD" | "OK" | "WARN" | "ERR")
                    || is_build_progress_line(normalized)
                    || normalized.starts_with("finished")
            }
            CommandDisplay::Test => false,
        }
    }

    fn shows_progress_line(self, normalized: &str) -> bool {
        match self {
            CommandDisplay::General | CommandDisplay::Build => {
                is_build_progress_line(normalized) || normalized.starts_with("finished")
            }
            CommandDisplay::Test => {
                is_test_build_progress_line(normalized) || normalized.starts_with("finished")
            }
        }
    }
}

#[derive(Debug)]
pub(crate) struct CommandFailure {
    label: String,
    status: String,
    log_path: PathBuf,
}

impl CommandFailure {
    fn new(label: impl Into<String>, status: impl Into<String>, log_path: PathBuf) -> Self {
        Self {
            label: label.into(),
            status: status.into(),
            log_path,
        }
    }

    pub(crate) fn log_path(&self) -> &Path {
        &self.log_path
    }
}

impl Display for CommandFailure {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} failed with {}; log: {}",
            self.label,
            self.status,
            display_path(&self.log_path)
        )
    }
}

impl Error for CommandFailure {}

/// Configures terminal output for the current xtask invocation.
pub(crate) fn init(ctx: &BuildContext, stream_subprocess: bool, no_banner: bool, plain: bool) {
    STREAM_SUBPROCESS.store(stream_subprocess, Ordering::Relaxed);
    NO_BANNER.store(no_banner, Ordering::Relaxed);
    let _ = COMMAND_LOG_DIR.set(ctx.command_logs_dir());
    let _ = WORKSPACE_ROOT.set(ctx.workspace_root().to_path_buf());

    let renderer = if should_use_inline(plain, stream_subprocess) {
        InlineRenderer::new(command_label()).ok().flatten()
    } else {
        None
    };

    if let Ok(mut slot) = inline_slot().lock() {
        *slot = renderer;
    }
}

/// Restores the terminal after bounded inline rendering.
pub(crate) fn finish(success: bool, summary: &str) {
    let Some(renderer) = inline_slot().lock().ok().and_then(|mut slot| slot.take()) else {
        print_final_status(success, summary);
        return;
    };
    renderer.finish(success, summary);
}

/// Returns whether subprocess output should stream live.
pub(crate) fn verbose() -> bool {
    STREAM_SUBPROCESS.load(Ordering::Relaxed)
}

/// Returns whether decorative banners are disabled.
pub(crate) fn no_banner() -> bool {
    NO_BANNER.load(Ordering::Relaxed)
}

/// Returns whether bounded inline rendering is active.
pub(crate) fn inline_active() -> bool {
    inline_slot()
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|_| ()))
        .is_some()
}

/// Prints a static banner for human-facing interactive flows.
pub(crate) fn banner(text: &str) {
    if no_banner() {
        return;
    }

    if with_inline(|renderer| {
        renderer.add_event(LineKind::Info, "INFO", text);
        renderer.render_now();
    })
    .is_some()
    {
        return;
    }

    if !use_terminal_decoration() {
        return;
    }

    let Ok(banner) = Banner::new(text) else {
        return;
    };

    println!(
        "{}",
        banner
            .style(BannerStyle::NeonCyber)
            .align(Align::Center)
            .padding(1)
            .render()
    );
}

/// Prints a major build phase header.
pub(crate) fn phase(title: &str) {
    if with_inline(|renderer| {
        renderer.visual = None;
        renderer.output.clear();
        renderer.footer_message = None;
        renderer.current_phase = Some(title.to_owned());
        renderer.add_event(LineKind::Run, "RUN", title);
        renderer.render_now();
    })
    .is_some()
    {
        return;
    }

    println!();
    println!(
        "{} {}",
        style("==").cyan().bold(),
        style(title).bold().cyan()
    );
}

/// Prints an informational key-value line.
pub(crate) fn detail(label: &str, value: impl Display) {
    let message = format!("{label}: {value}");
    if with_inline(|renderer| {
        renderer.add_event(LineKind::Info, "INFO", &message);
        renderer.render_now();
    })
    .is_some()
    {
        return;
    }

    println!(
        "   {} {} {}",
        style("INFO").dim(),
        style(format!("{label}:")).dim(),
        value
    );
}

/// Prints an important filesystem path.
pub(crate) fn path(label: &str, path: &Path) {
    let path = display_path(path);
    let message = format!("{label}: {path}");
    if with_inline(|renderer| {
        renderer.add_event(LineKind::Path, "PATH", &message);
        renderer.render_now();
    })
    .is_some()
    {
        return;
    }

    println!(
        "   {} {} {}",
        style("PATH").cyan().bold(),
        style(format!("{label}:")).dim(),
        style(path).cyan()
    );
}

/// Prints a short step that does not need spinner lifecycle tracking.
pub(crate) fn step(message: impl Display) {
    let message = message.to_string();
    if with_inline(|renderer| {
        renderer.add_event(LineKind::Run, "RUN", &message);
        renderer.render_now();
    })
    .is_some()
    {
        return;
    }

    println!("   {} {message}", style("RUN").blue().bold());
}

/// Prints a successful status line.
pub(crate) fn success(message: impl Display) {
    let message = message.to_string();
    if with_inline(|renderer| {
        renderer.add_event(LineKind::Ok, "OK", &message);
        renderer.render_now();
    })
    .is_some()
    {
        return;
    }

    println!(
        "   {} {}",
        style("OK").green().bold(),
        style(message).green()
    );
}

/// Prints an artifact path.
pub(crate) fn artifact(path: &Path) {
    let message = format!("Path: {}", path.display());
    if with_inline(|renderer| {
        renderer.add_event(LineKind::Artifact, "ARTIFACT", &message);
        renderer.render_now();
    })
    .is_some()
    {
        return;
    }

    println!(
        "   {} {} {}",
        style("ARTIFACT").magenta().bold(),
        style("Path:").dim(),
        path.display()
    );
}

/// Prints a warning line.
pub(crate) fn warning(message: impl Display) {
    let message = message.to_string();
    if with_inline(|renderer| {
        renderer.add_event(LineKind::Warn, "WARN", &message);
        renderer.render_now();
    })
    .is_some()
    {
        return;
    }

    eprintln!(
        "   {} {}",
        style("WARN").yellow().bold(),
        style(message).yellow()
    );
}

/// Shows a bounded visual frame, used by setup splash animation.
pub(crate) fn visual_frame(lines: Vec<String>, frame: usize) -> bool {
    with_inline(|renderer| {
        renderer.visual = Some(InlineVisual { lines, frame });
        renderer.render_now();
    })
    .is_some()
}

/// Clears the bounded visual frame.
pub(crate) fn clear_visual() {
    let _ = with_inline(|renderer| {
        renderer.visual = None;
        renderer.render_now();
    });
}

/// Reads a select prompt inside the bounded inline viewport.
pub(crate) fn prompt_select(
    message: &str,
    options: &[String],
    default: usize,
) -> Result<Option<usize>> {
    if !inline_active() {
        return Ok(None);
    }
    if options.is_empty() {
        return Err(anyhow!("prompt options cannot be empty"));
    }

    let _raw = RawModeGuard::enter()?;
    let mut selected = default.min(options.len().saturating_sub(1));
    set_prompt(Some(PromptState {
        message: message.to_owned(),
        options: options.to_vec(),
        selected,
    }));

    loop {
        let Event::Key(key) = event::read()? else {
            continue;
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }

        match key.code {
            KeyCode::Up | KeyCode::Char('k') => {
                selected = selected.saturating_sub(1);
            }
            KeyCode::Down | KeyCode::Char('j') => {
                selected = (selected + 1).min(options.len() - 1);
            }
            KeyCode::Home => {
                selected = 0;
            }
            KeyCode::End => {
                selected = options.len() - 1;
            }
            KeyCode::Enter => {
                set_prompt(None);
                detail(message, &options[selected]);
                return Ok(Some(selected));
            }
            KeyCode::Esc => {
                set_prompt(None);
                return Err(anyhow!("prompt cancelled"));
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                set_prompt(None);
                return Err(anyhow!("prompt cancelled"));
            }
            _ => {}
        }

        set_prompt(Some(PromptState {
            message: message.to_owned(),
            options: options.to_vec(),
            selected,
        }));
    }
}

/// Reads a yes/no prompt inside the bounded inline viewport.
pub(crate) fn prompt_confirm(message: &str, default: bool) -> Result<Option<bool>> {
    let options = vec!["Yes".to_owned(), "No".to_owned()];
    let default_index = if default { 0 } else { 1 };
    let Some(index) = prompt_select(message, &options, default_index)? else {
        return Ok(None);
    };
    Ok(Some(index == 0))
}

/// Runs an external command with spinner, clean log capture, and elapsed-time reporting.
pub(crate) fn run_command(label: impl Into<String>, command: Cmd<'_>) -> Result<()> {
    run_command_with_display(label, command, CommandDisplay::General)
}

/// Runs a build subprocess with compact progress rendering.
pub(crate) fn run_build_command(label: impl Into<String>, command: Cmd<'_>) -> Result<()> {
    run_command_with_display(label, command, CommandDisplay::Build)
}

/// Runs a test subprocess without exposing raw runner output by default.
pub(crate) fn run_test_command(label: impl Into<String>, command: Cmd<'_>) -> Result<()> {
    run_command_with_display(label, command, CommandDisplay::Test)
}

fn run_command_with_display(
    label: impl Into<String>,
    command: Cmd<'_>,
    display: CommandDisplay,
) -> Result<()> {
    let label = label.into();
    if verbose() {
        return run_command_verbose(label, command);
    }

    if inline_active() {
        return run_command_inline(label, command, display);
    }

    let step = TimedStep::start(label.clone(), true);
    let output = command
        .quiet()
        .ignore_status()
        .output()
        .with_context(|| format!("{label} failed to start"))?;

    if output.status.success() {
        step.finish_success();
        return Ok(());
    }

    step.finish_error();
    let log_path = write_command_log(&label, &output)?;
    print_command_failure(&label, &output, &log_path);
    Err(CommandFailure::new(label, command_status_label(&output), log_path).into())
}

/// Runs a long-lived subprocess while keeping logs bounded in the inline viewport.
pub(crate) fn run_long_command(label: impl Into<String>, command: Cmd<'_>) -> Result<()> {
    let label = label.into();
    if verbose() || !inline_active() {
        return run_command_verbose(label, command);
    }

    run_command_inline(label, command, CommandDisplay::General)
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

fn run_command_inline(label: String, command: Cmd<'_>, display: CommandDisplay) -> Result<()> {
    let started_at = Instant::now();
    let (log_path, log_file) = open_command_log(&label)?;
    let log_file = Arc::new(Mutex::new(log_file));
    start_inline_step(&label, &log_path);

    let mut process: std::process::Command = command.quiet().into();
    apply_rich_terminal_env(&mut process);
    if should_use_pty(&label, &process) {
        return run_command_pty(label, started_at, process, log_path, log_file, display);
    }

    run_command_piped(label, started_at, process, log_path, log_file, display)
}

fn run_command_piped(
    label: String,
    started_at: Instant,
    mut process: std::process::Command,
    log_path: PathBuf,
    log_file: Arc<Mutex<File>>,
    display: CommandDisplay,
) -> Result<()> {
    process
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(error) => {
            finish_inline_step(&label, started_at, false);
            print_status_failure(&label, None, &log_path);
            return Err(error).with_context(|| format!("{label} failed to start"));
        }
    };

    let stdout_thread = child.stdout.take().map(|stream| {
        spawn_output_reader(OutputStream::Stdout, stream, Arc::clone(&log_file), display)
    });
    let stderr_thread = child.stderr.take().map(|stream| {
        spawn_output_reader(OutputStream::Stderr, stream, Arc::clone(&log_file), display)
    });
    let ticker = InlineTicker::start();

    let status = match child.wait() {
        Ok(status) => status,
        Err(error) => {
            drop(ticker);
            finish_inline_step(&label, started_at, false);
            set_inline_footer(format!(
                "ERR {label} wait failed | log: {}",
                display_path(&log_path)
            ));
            return Err(error)
                .with_context(|| format!("{label} failed while waiting for process exit"));
        }
    };
    drop(ticker);
    let reader_result =
        join_output_reader(stdout_thread).and_then(|_| join_output_reader(stderr_thread));
    if let Err(error) = reader_result {
        finish_inline_step(&label, started_at, false);
        error_event(format!("{label} output capture failed: {error:#}"));
        set_inline_footer(format!(
            "ERR {label} output capture failed | log: {}",
            display_path(&log_path)
        ));
        return Err(error);
    }

    if let Ok(mut log) = log_file.lock() {
        let _ = writeln!(log);
        let _ = writeln!(log, "status: {status}");
    }

    if status.success() {
        finish_inline_step(&label, started_at, true);
        return Ok(());
    }

    finish_inline_step(&label, started_at, false);
    print_status_failure(&label, Some(&status), &log_path);
    Err(CommandFailure::new(label, exit_status_label(&status), log_path).into())
}

fn run_command_pty(
    label: String,
    started_at: Instant,
    process: std::process::Command,
    log_path: PathBuf,
    log_file: Arc<Mutex<File>>,
    display: CommandDisplay,
) -> Result<()> {
    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(inline_pty_size())
        .with_context(|| format!("{label} failed to open pty"))?;
    let master = pair.master;
    let slave = pair.slave;
    let reader = master
        .try_clone_reader()
        .with_context(|| format!("{label} failed to open pty reader"))?;
    let mut child = slave
        .spawn_command(command_to_pty_builder(&process))
        .with_context(|| format!("{label} failed to start"))?;
    drop(slave);
    let writer = master.take_writer().ok();

    let output_thread = Some(spawn_output_reader(
        OutputStream::Pty,
        reader,
        Arc::clone(&log_file),
        display,
    ));
    let ticker = InlineTicker::start();
    let status = match child.wait() {
        Ok(status) => status,
        Err(error) => {
            drop(ticker);
            finish_inline_step(&label, started_at, false);
            set_inline_footer(format!(
                "ERR {label} wait failed | log: {}",
                display_path(&log_path)
            ));
            return Err(error)
                .with_context(|| format!("{label} failed while waiting for process exit"));
        }
    };
    drop(ticker);
    drop(writer);
    drop(master);

    if let Err(error) = join_output_reader(output_thread) {
        finish_inline_step(&label, started_at, false);
        error_event(format!("{label} output capture failed: {error:#}"));
        set_inline_footer(format!(
            "ERR {label} output capture failed | log: {}",
            display_path(&log_path)
        ));
        return Err(error);
    }

    if let Ok(mut log) = log_file.lock() {
        let _ = writeln!(log);
        let _ = writeln!(log, "status: {}", pty_status_label(&status));
    }

    if status.success() {
        finish_inline_step(&label, started_at, true);
        return Ok(());
    }

    finish_inline_step(&label, started_at, false);
    print_pty_status_failure(&label, &status, &log_path);
    Err(CommandFailure::new(label, pty_status_label(&status), log_path).into())
}

fn apply_rich_terminal_env(process: &mut std::process::Command) {
    let width = inline_width().unwrap_or(100).saturating_sub(16).max(40);
    process
        .env("TERM", "xterm-256color")
        .env("COLORTERM", "truecolor")
        .env("CLICOLOR_FORCE", "1")
        .env("FORCE_COLOR", "3")
        .env("CARGO_TERM_COLOR", "always")
        .env("CARGO_TERM_PROGRESS_WHEN", "always")
        .env("CARGO_TERM_PROGRESS_WIDTH", width.to_string());
}

fn should_use_pty(label: &str, process: &std::process::Command) -> bool {
    if !use_terminal_decoration() {
        return false;
    }
    if cfg!(windows) {
        return false;
    }

    let label = label.to_ascii_lowercase();
    if [
        "build", "compil", "link", "maturin", "napi", "cargo", "package", "stage",
    ]
    .iter()
    .any(|needle| label.contains(needle))
    {
        return true;
    }

    process
        .get_program()
        .to_string_lossy()
        .to_ascii_lowercase()
        .contains("cargo")
        || process
            .get_args()
            .any(|arg| arg.to_string_lossy().to_ascii_lowercase().contains("napi"))
}

fn inline_width() -> Option<u16> {
    inline_slot()
        .lock()
        .ok()
        .and_then(|slot| slot.as_ref().map(|renderer| renderer.width))
}

fn inline_pty_size() -> PtySize {
    let (cols, rows) = inline_slot()
        .lock()
        .ok()
        .and_then(|slot| {
            slot.as_ref()
                .map(|renderer| (renderer.width, renderer.height))
        })
        .unwrap_or((100, 24));

    PtySize {
        rows: rows.max(12),
        cols: cols.max(60),
        pixel_width: 0,
        pixel_height: 0,
    }
}

fn command_to_pty_builder(process: &std::process::Command) -> CommandBuilder {
    let mut argv = Vec::new();
    argv.push(process.get_program().to_os_string());
    argv.extend(process.get_args().map(|arg| arg.to_os_string()));

    let mut builder = CommandBuilder::from_argv(argv);
    if let Some(cwd) = process.get_current_dir() {
        builder.cwd(cwd.as_os_str());
    }
    for (key, value) in process.get_envs() {
        match value {
            Some(value) => builder.env(key, value),
            None => builder.env_remove(key),
        }
    }
    builder
}

fn pty_status_label(status: &portable_pty::ExitStatus) -> String {
    if let Some(signal) = status.signal() {
        format!("terminated by {signal}")
    } else {
        format!("exit code {}", status.exit_code())
    }
}

fn display_path(path: &Path) -> String {
    let relative = WORKSPACE_ROOT
        .get()
        .and_then(|root| path.strip_prefix(root).ok());
    let path = relative.unwrap_or(path);
    path.display().to_string()
}

fn run_command_verbose(label: String, command: Cmd<'_>) -> Result<()> {
    let step = TimedStep::start(label.clone(), false);
    let result = command.quiet().run();

    match result {
        Ok(()) => {
            step.finish_success();
            Ok(())
        }
        Err(error) => {
            step.finish_error();
            Err(error).with_context(|| format!("{label} failed"))
        }
    }
}

fn print_final_status(success: bool, summary: &str) {
    if success {
        println!(
            "   {} {} complete",
            style("OK").green().bold(),
            style(summary).green()
        );
    } else {
        eprintln!(
            "   {} {} failed",
            style("ERR").red().bold(),
            style(summary).red()
        );
    }
}

fn write_command_log(label: &str, output: &Output) -> Result<PathBuf> {
    let (path, mut log) = open_command_log(label)?;
    let _ = writeln!(log, "status: {}", output.status);
    let _ = writeln!(log);
    let _ = writeln!(log, "stdout:");
    let _ = write!(log, "{}", String::from_utf8_lossy(&output.stdout));
    let _ = writeln!(log);
    let _ = writeln!(log, "stderr:");
    let _ = write!(log, "{}", String::from_utf8_lossy(&output.stderr));
    Ok(path)
}

fn open_command_log(label: &str) -> Result<(PathBuf, File)> {
    let log_dir = COMMAND_LOG_DIR
        .get()
        .cloned()
        .unwrap_or_else(|| PathBuf::from(".build").join("logs"));
    fs::create_dir_all(&log_dir)
        .with_context(|| format!("failed to create {}", log_dir.display()))?;

    let path = log_dir.join(command_log_file_name(label));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&path)
        .with_context(|| format!("failed to write {}", path.display()))?;
    let _ = writeln!(file, "step: {label}");
    Ok((path, file))
}

fn command_log_file_name(label: &str) -> String {
    let elapsed_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0);
    let index = COMMAND_LOG_COUNTER.fetch_add(1, Ordering::Relaxed);
    let process_id = std::process::id();
    format!(
        "{elapsed_ms}-{process_id}-{index}-{}.log",
        sanitize_log_label(label)
    )
}

fn sanitize_log_label(label: &str) -> String {
    let mut sanitized = String::with_capacity(label.len());
    let mut last_was_dash = false;

    for character in label.chars() {
        if character.is_ascii_alphanumeric() {
            sanitized.push(character.to_ascii_lowercase());
            last_was_dash = false;
        } else if !last_was_dash {
            sanitized.push('-');
            last_was_dash = true;
        }
    }

    let sanitized = sanitized.trim_matches('-');
    if sanitized.is_empty() {
        "command".to_owned()
    } else {
        sanitized.to_owned()
    }
}

fn print_command_failure(label: &str, output: &Output, log_path: &Path) {
    if inline_active() {
        let status = command_status_label(output);
        error_event(format!("{label} failed with {status}"));
        path("Log", log_path);
        return;
    }

    eprintln!(
        "   {} {} failed with {}",
        style("ERR").red().bold(),
        label,
        command_status_label(output)
    );
    eprintln!(
        "   {} {}",
        style("PATH").cyan().bold(),
        style(display_path(log_path)).cyan()
    );
}

fn print_status_failure(label: &str, status: Option<&ExitStatus>, log_path: &Path) {
    let status = status
        .map(exit_status_label)
        .unwrap_or_else(|| "failed to start".to_owned());
    if inline_active() {
        error_event(format!("{label} failed with {status}"));
        set_inline_footer(format!(
            "ERR {label} failed with {status} | log: {}",
            display_path(log_path)
        ));
        path("Log", log_path);
        return;
    }

    eprintln!(
        "   {} {} failed with {}",
        style("ERR").red().bold(),
        label,
        status
    );
    eprintln!(
        "   {} {}",
        style("PATH").cyan().bold(),
        style(display_path(log_path)).cyan()
    );
}

fn print_pty_status_failure(label: &str, status: &portable_pty::ExitStatus, log_path: &Path) {
    let status = pty_status_label(status);
    if inline_active() {
        error_event(format!("{label} failed with {status}"));
        set_inline_footer(format!(
            "ERR {label} failed with {status} | log: {}",
            display_path(log_path)
        ));
        path("Log", log_path);
        return;
    }

    eprintln!(
        "   {} {} failed with {}",
        style("ERR").red().bold(),
        label,
        status
    );
    eprintln!(
        "   {} {}",
        style("PATH").cyan().bold(),
        style(display_path(log_path)).cyan()
    );
}

#[cfg(test)]
fn tail_output_lines(output: &Output, max_lines: usize) -> Vec<String> {
    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&output.stdout));
    if !output.stdout.is_empty() && !output.stderr.is_empty() {
        combined.push('\n');
    }
    combined.push_str(&String::from_utf8_lossy(&output.stderr));

    let mut lines = combined
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .rev()
        .take(max_lines)
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    lines.reverse();
    lines
}

fn command_status_label(output: &Output) -> String {
    exit_status_label(&output.status)
}

fn exit_status_label(status: &ExitStatus) -> String {
    status
        .code()
        .map(|code| format!("exit code {code}"))
        .unwrap_or_else(|| "terminated by signal".to_owned())
}

#[derive(Clone, Copy)]
enum OutputStream {
    Stdout,
    Stderr,
    Pty,
}

impl OutputStream {
    fn name(self) -> &'static str {
        match self {
            OutputStream::Stdout => "stdout",
            OutputStream::Stderr => "stderr",
            OutputStream::Pty => "pty",
        }
    }

    fn label(self) -> &'static str {
        match self {
            OutputStream::Stdout => "OUT",
            OutputStream::Stderr => "LOG",
            OutputStream::Pty => "LOG",
        }
    }

    fn kind(self) -> LineKind {
        match self {
            OutputStream::Stdout => LineKind::Output,
            OutputStream::Stderr => LineKind::Output,
            OutputStream::Pty => LineKind::Output,
        }
    }

    fn classify(self, line: &str) -> (&'static str, LineKind) {
        let normalized = line.trim_start().to_ascii_lowercase();
        if normalized.starts_with("error") || normalized.contains("error:") {
            ("ERR", LineKind::Err)
        } else if normalized.starts_with("warning") || normalized.contains("warning:") {
            ("WARN", LineKind::Warn)
        } else if normalized.starts_with("finished") {
            ("OK", LineKind::Ok)
        } else if is_build_progress_line(&normalized) {
            ("BUILD", LineKind::Run)
        } else {
            (self.label(), self.kind())
        }
    }
}

fn is_build_progress_line(normalized: &str) -> bool {
    [
        "building",
        "compiling",
        "checking",
        "linking",
        "running",
        "packaging",
        "staging",
        "copying",
        "downloading",
        "installing",
        "updating",
        "fetching",
        "resolving",
    ]
    .iter()
    .any(|prefix| normalized.starts_with(prefix))
}

fn is_test_build_progress_line(normalized: &str) -> bool {
    [
        "building",
        "compiling",
        "checking",
        "linking",
        "packaging",
        "staging",
        "copying",
        "downloading",
        "installing",
        "updating",
        "fetching",
        "resolving",
    ]
    .iter()
    .any(|prefix| normalized.starts_with(prefix))
}

fn spawn_output_reader<R>(
    output_stream: OutputStream,
    stream: R,
    log_file: Arc<Mutex<File>>,
    display: CommandDisplay,
) -> JoinHandle<Result<()>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let stream_name = output_stream.name();
        let mut reader = BufReader::new(stream);
        let mut chunk = [0u8; 4096];
        let mut pending = Vec::new();
        let mut pending_cr = false;

        loop {
            let read = reader
                .read(&mut chunk)
                .with_context(|| format!("failed to read {stream_name}"))?;
            if read == 0 {
                break;
            }

            if let Ok(mut log) = log_file.lock() {
                let _ = write!(log, "[{stream_name}] ");
                let _ = log.write_all(&chunk[..read]);
                let _ = writeln!(log);
            }

            for byte in &chunk[..read] {
                if pending_cr {
                    if *byte == b'\n' {
                        emit_subprocess_line(&pending, output_stream, display);
                        pending.clear();
                        pending_cr = false;
                        continue;
                    }

                    emit_subprocess_progress(&pending, output_stream, display);
                    pending.clear();
                    pending_cr = false;
                }

                if *byte == b'\r' {
                    pending_cr = true;
                } else if *byte == b'\n' {
                    emit_subprocess_line(&pending, output_stream, display);
                    pending.clear();
                } else {
                    pending.push(*byte);
                }

                if pending.len() >= 4096 {
                    emit_subprocess_line(&pending, output_stream, display);
                    pending.clear();
                }
            }
        }

        if pending_cr {
            emit_subprocess_progress(&pending, output_stream, display);
        } else {
            emit_subprocess_line(&pending, output_stream, display);
        }
        Ok(())
    })
}

fn emit_subprocess_line(bytes: &[u8], stream: OutputStream, display: CommandDisplay) {
    let line = String::from_utf8_lossy(bytes);
    let line = line.trim_end();
    if !plain_stream_text(line).trim().is_empty() {
        subprocess_event(&line, stream, display);
    }
}

fn emit_subprocess_progress(bytes: &[u8], stream: OutputStream, display: CommandDisplay) {
    let line = String::from_utf8_lossy(bytes);
    let line = line.trim_end();
    if !plain_stream_text(line).trim().is_empty() {
        subprocess_progress_event(&line, stream, display);
    }
}

fn plain_stream_text(text: &str) -> String {
    let mut sanitized = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(character) = chars.next() {
        if character == '\u{1b}' {
            if matches!(chars.peek(), Some('[')) {
                let _ = chars.next();
                for next in chars.by_ref() {
                    if ('@'..='~').contains(&next) {
                        break;
                    }
                }
            } else if matches!(chars.peek(), Some(']')) {
                let _ = chars.next();
                while let Some(next) = chars.next() {
                    if next == '\u{7}' {
                        break;
                    }
                    if next == '\u{1b}' && matches!(chars.peek(), Some('\\')) {
                        let _ = chars.next();
                        break;
                    }
                }
            } else {
                let _ = chars.next();
            }
        } else if character == '\t' {
            sanitized.push(' ');
        } else if !character.is_control() {
            sanitized.push(character);
        }
    }

    sanitized
}

fn join_output_reader(handle: Option<JoinHandle<Result<()>>>) -> Result<()> {
    let Some(handle) = handle else {
        return Ok(());
    };
    match handle.join() {
        Ok(result) => result,
        Err(_) => Err(anyhow!("subprocess output reader panicked")),
    }
}

fn start_inline_step(label: &str, log_path: &Path) {
    let _ = with_inline(|renderer| {
        renderer.active = Some(ActiveStep {
            label: label.to_owned(),
            started_at: Instant::now(),
            frame: 0,
        });
        renderer.output.clear();
        renderer.progress = None;
        renderer.current_log_path = Some(log_path.to_path_buf());
        renderer.footer_message = Some(format!("Log: {}", display_path(log_path)));
        renderer.add_event(LineKind::Run, "RUN", label);
        renderer.render_now();
    });
}

fn finish_inline_step(label: &str, started_at: Instant, ok: bool) {
    let _ = with_inline(|renderer| {
        renderer.active = None;
        let message = format!("{label} ({})", elapsed(started_at.elapsed()));
        if ok {
            renderer.add_event(LineKind::Ok, "OK", &message);
            renderer.footer_message = Some(match renderer.current_log_path.as_ref() {
                Some(log_path) => format!("OK {message} | log: {}", display_path(log_path)),
                None => format!("OK {message}"),
            });
        } else {
            renderer.add_event(LineKind::Err, "ERR", &message);
            renderer.footer_message = Some(match renderer.current_log_path.as_ref() {
                Some(log_path) => format!("ERR {message} | log: {}", display_path(log_path)),
                None => format!("ERR {message}"),
            });
        }
        renderer.render_now();
    });
}

fn tick_inline_step() {
    let _ = with_inline(|renderer| {
        if let Some(active) = renderer.active.as_mut() {
            active.frame = active.frame.wrapping_add(1);
            renderer.render_throttled();
        }
    });
}

fn subprocess_event(line: &str, stream: OutputStream, display: CommandDisplay) {
    let _ = with_inline(|renderer| {
        let plain = plain_stream_text(line);
        let (label, kind) = stream.classify(&plain);
        let normalized = plain.trim_start().to_ascii_lowercase();
        if display.shows_progress_line(&normalized) {
            renderer.set_progress(kind, label, line);
        }
        if display.shows_output_line(label, &normalized) {
            renderer.add_output(kind, label, line);
        }
        renderer.render_throttled();
    });
}

fn subprocess_progress_event(line: &str, stream: OutputStream, display: CommandDisplay) {
    let _ = with_inline(|renderer| {
        let plain = plain_stream_text(line);
        let normalized = plain.trim_start().to_ascii_lowercase();
        if !display.shows_progress_line(&normalized) {
            return;
        }
        let (label, kind) = stream.classify(&plain);
        renderer.set_progress(kind, label, line);
        renderer.render_throttled();
    });
}

fn error_event(message: impl Display) {
    let message = message.to_string();
    let _ = with_inline(|renderer| {
        renderer.add_event(LineKind::Err, "ERR", &message);
        renderer.render_now();
    });
}

fn set_prompt(prompt: Option<PromptState>) {
    let _ = with_inline(|renderer| {
        renderer.prompt = prompt;
        renderer.render_now();
    });
}

fn set_inline_footer(message: impl Into<String>) {
    let message = message.into();
    let _ = with_inline(|renderer| {
        renderer.footer_message = Some(message);
        renderer.render_now();
    });
}

fn inline_slot() -> &'static Mutex<Option<InlineRenderer>> {
    INLINE_RENDERER.get_or_init(|| Mutex::new(None))
}

fn with_inline<R>(f: impl FnOnce(&mut InlineRenderer) -> R) -> Option<R> {
    let mut slot = inline_slot().lock().ok()?;
    let renderer = slot.as_mut()?;
    Some(f(renderer))
}

fn should_use_inline(plain: bool, verbose: bool) -> bool {
    !plain && !verbose && use_terminal_decoration()
}

fn use_spinner() -> bool {
    use_terminal_decoration() && !verbose()
}

fn use_terminal_decoration() -> bool {
    Term::stdout().is_term() && env::var_os("CI").is_none() && env::var_os("NO_COLOR").is_none()
}

fn command_label() -> String {
    let args = env::args().skip(1).collect::<Vec<_>>();
    if args.is_empty() {
        "xtask".to_owned()
    } else {
        args.join(" ")
    }
}

struct InlineTicker {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl InlineTicker {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(100));
                tick_inline_step();
            }
        });

        Self {
            stop,
            handle: Some(handle),
        }
    }
}

impl Drop for InlineTicker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

struct RawModeGuard;

impl RawModeGuard {
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
    }
}

struct TimedStep {
    label: String,
    started_at: Instant,
    spinner: Option<ProgressBar>,
}

impl TimedStep {
    fn start(label: impl Into<String>, allow_spinner: bool) -> Self {
        let label = label.into();
        let spinner = if allow_spinner && use_spinner() {
            let spinner = ProgressBar::new_spinner();
            if let Ok(style) = ProgressStyle::with_template("{spinner:.cyan} {msg}") {
                spinner.set_style(style.tick_strings(&[".  ", ".. ", "...", " ..", "  .", "   "]));
            }
            spinner.set_message(label.clone());
            spinner.enable_steady_tick(Duration::from_millis(100));
            Some(spinner)
        } else {
            println!("   {} {label}", style("RUN").blue().bold());
            None
        };

        Self {
            label,
            started_at: Instant::now(),
            spinner,
        }
    }

    fn finish_success(self) {
        let message = format!(
            "{} {} {}",
            style("OK").green().bold(),
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
            "{} {} {}",
            style("ERR").red().bold(),
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

struct InlineRenderer {
    command_label: String,
    origin_y: u16,
    height: u16,
    width: u16,
    started_at: Instant,
    current_phase: Option<String>,
    active: Option<ActiveStep>,
    activity: VecDeque<ActivityLine>,
    output: VecDeque<OutputLine>,
    progress: Option<OutputLine>,
    current_log_path: Option<PathBuf>,
    footer_message: Option<String>,
    final_status: Option<FinalStatus>,
    visual: Option<InlineVisual>,
    prompt: Option<PromptState>,
    previous_rows: Vec<Option<RenderedRow>>,
    last_rendered_at: Instant,
}

impl InlineRenderer {
    fn new(command_label: String) -> Result<Option<Self>> {
        let (columns, rows) = terminal::size()?;
        if columns < INLINE_MIN_COLUMNS {
            return Ok(None);
        }

        let height = INLINE_DEFAULT_LINES.min(rows.saturating_sub(2));
        if height < INLINE_MIN_LINES {
            return Ok(None);
        }

        let mut stdout = std::io::stdout();
        for _ in 0..height {
            writeln!(stdout)?;
        }
        execute!(stdout, MoveToPreviousLine(height))?;
        let (_, origin_y) = position()?;
        let _ = execute!(stdout, DisableLineWrap, Hide);
        let previous_rows = vec![None; height as usize];

        let mut renderer = Self {
            command_label,
            origin_y,
            height,
            width: columns,
            started_at: Instant::now(),
            current_phase: None,
            active: None,
            activity: VecDeque::new(),
            output: VecDeque::new(),
            progress: None,
            current_log_path: None,
            footer_message: None,
            final_status: None,
            visual: None,
            prompt: None,
            previous_rows,
            last_rendered_at: Instant::now(),
        };
        renderer.render_now();
        Ok(Some(renderer))
    }

    fn finish(mut self, success: bool, summary: &str) {
        self.active = None;
        self.prompt = None;
        self.visual = None;
        self.final_status = Some(FinalStatus {
            success,
            summary: summary.to_owned(),
        });
        let label = if success { "OK" } else { "ERR" };
        let kind = if success { LineKind::Ok } else { LineKind::Err };
        let message = if success {
            format!("{summary} complete")
        } else {
            format!("{summary} failed")
        };
        self.add_event(kind, label, &message);
        self.footer_message = Some(format!("{label} {message}"));
        self.render_now();

        let mut stdout = std::io::stdout();
        let _ = execute!(
            stdout,
            MoveTo(0, self.origin_y.saturating_add(self.height)),
            ResetColor,
            SetAttribute(Attribute::Reset),
            EnableLineWrap,
            Show
        );
    }

    fn add_event(&mut self, kind: LineKind, label: &'static str, text: &str) {
        self.activity.push_back(ActivityLine {
            kind,
            label,
            text: text.to_owned(),
        });
        while self.activity.len() > MAX_ACTIVITY_LINES {
            let _ = self.activity.pop_front();
        }
    }

    fn add_output(&mut self, kind: LineKind, label: &'static str, text: &str) {
        self.output.push_back(OutputLine {
            kind,
            label,
            segments: parse_ansi_segments(text),
        });
        while self.output.len() > MAX_OUTPUT_LINES {
            let _ = self.output.pop_front();
        }
    }

    fn set_progress(&mut self, kind: LineKind, label: &'static str, text: &str) {
        self.progress = Some(OutputLine {
            kind,
            label,
            segments: parse_ansi_segments(text),
        });
    }

    fn render_throttled(&mut self) {
        if self.last_rendered_at.elapsed() >= INLINE_MIN_RENDER_INTERVAL {
            self.render_now();
        }
    }

    fn render_now(&mut self) {
        let Ok((columns, _)) = terminal::size() else {
            return;
        };
        let resized = columns != self.width;
        self.width = columns;
        if resized || self.previous_rows.len() != self.height as usize {
            self.previous_rows = vec![None; self.height as usize];
        }

        let frame = self.build_frame();
        let mut stdout = std::io::stdout();
        for (row, rendered) in frame.into_iter().enumerate() {
            let changed = self.previous_rows.get(row).and_then(Option::as_ref) != Some(&rendered);
            if !changed {
                continue;
            }

            self.draw_rendered_row(&mut stdout, row as u16, &rendered, resized);
            if let Some(previous) = self.previous_rows.get_mut(row) {
                *previous = Some(rendered);
            }
        }

        let _ = queue!(
            stdout,
            ResetColor,
            SetAttribute(Attribute::Reset),
            MoveTo(0, self.origin_y.saturating_add(self.height))
        );
        let _ = stdout.flush();
        self.last_rendered_at = Instant::now();
    }

    fn build_frame(&self) -> Vec<RenderedRow> {
        let mut rows = vec![RenderedRow::blank(); self.height as usize];
        self.render_header(&mut rows);
        self.render_status(&mut rows);

        if let Some(prompt) = &self.prompt {
            self.render_prompt(&mut rows, prompt);
        } else if let Some(visual) = &self.visual {
            self.render_visual(&mut rows, visual);
        } else {
            self.render_activity(&mut rows);
        }

        self.render_footer(&mut rows);
        rows
    }

    fn render_header(&self, rows: &mut [RenderedRow]) {
        let elapsed = elapsed(self.started_at.elapsed());
        let title = format!(" SIPP | {} | {}", self.command_label, elapsed);
        self.put_line(rows, 0, &title, Color::Cyan, true);
        self.put_line(
            rows,
            1,
            &"-".repeat(self.width as usize),
            Color::DarkGrey,
            false,
        );
    }

    fn render_status(&self, rows: &mut [RenderedRow]) {
        let phase = self
            .current_phase
            .as_deref()
            .unwrap_or("Starting developer workflow");
        self.put_labeled_line(rows, 2, "RUN", phase, LineKind::Run);

        if let Some(step) = self.active.as_ref() {
            let active = format!(
                "{} {}",
                spinner_frame(step.frame),
                active_step_message(&step.label, step.started_at.elapsed())
            );
            self.put_labeled_line(rows, 3, "RUN", &active, LineKind::Run);
        } else if let Some(status) = self.final_status.as_ref() {
            let label = if status.success { "OK" } else { "ERR" };
            let kind = if status.success {
                LineKind::Ok
            } else {
                LineKind::Err
            };
            let text = if status.success {
                format!("{} complete", status.summary.as_str())
            } else {
                format!("{} failed", status.summary.as_str())
            };
            self.put_labeled_line(rows, 3, label, &text, kind);
        } else {
            self.put_labeled_line(rows, 3, "OK", "Idle", LineKind::Ok);
        }

        self.put_line(
            rows,
            4,
            &"-".repeat(self.width as usize),
            Color::DarkGrey,
            false,
        );
    }

    fn render_activity(&self, rows: &mut [RenderedRow]) {
        let mut body_start = 5;
        let body_end = self.height.saturating_sub(2);
        if body_end <= body_start {
            return;
        }

        if let Some(progress) = self.progress.as_ref() {
            self.put_styled_labeled_line(
                rows,
                body_start,
                progress.label,
                &progress.segments,
                progress.kind,
            );
            body_start += 1;
        }

        let body_lines = body_end.saturating_sub(body_start) as usize;
        if body_lines == 0 {
            return;
        }

        if self.output.is_empty() {
            self.render_activity_history(rows, body_start, body_lines);
            return;
        }

        let start = self.output.len().saturating_sub(body_lines);
        for (offset, line) in self.output.iter().skip(start).enumerate() {
            self.put_styled_labeled_line(
                rows,
                body_start + offset as u16,
                line.label,
                &line.segments,
                line.kind,
            );
        }
    }

    fn render_activity_history(&self, rows: &mut [RenderedRow], mut row: u16, body_lines: usize) {
        let mut remaining = body_lines;
        let active_label = self.active.as_ref().map(|step| step.label.as_str());
        if let Some(step) = self.active.as_ref() {
            self.put_labeled_line(
                rows,
                row,
                "RUN",
                &active_step_message(&step.label, step.started_at.elapsed()),
                LineKind::Run,
            );
            row += 1;
            remaining = remaining.saturating_sub(1);
        }
        if remaining == 0 {
            return;
        }

        let lines = self
            .activity
            .iter()
            .filter(|line| {
                !matches!(
                    active_label,
                    Some(active) if line.label == "RUN" && line.text == active
                )
            })
            .collect::<Vec<_>>();
        let start = lines.len().saturating_sub(remaining);
        for (offset, line) in lines.into_iter().skip(start).enumerate() {
            self.put_labeled_line(rows, row + offset as u16, line.label, &line.text, line.kind);
        }
    }

    fn render_visual(&self, rows: &mut [RenderedRow], visual: &InlineVisual) {
        let body_start = 5;
        let body_end = self.height.saturating_sub(2);
        let body_lines = body_end.saturating_sub(body_start) as usize;
        if body_lines == 0 {
            return;
        }

        let visible = visual.lines.iter().take(body_lines).collect::<Vec<_>>();
        let block_width = visible
            .iter()
            .map(|line| measure_text_width(line))
            .max()
            .unwrap_or(0);
        let left_padding = (self.width as usize).saturating_sub(block_width) / 2;
        let top_padding = body_lines.saturating_sub(visible.len()) / 2;
        for (offset, line) in visible.into_iter().enumerate() {
            let mut segments = Vec::new();
            if left_padding > 0 {
                segments.push(RowSegment::new(
                    " ".repeat(left_padding),
                    Color::White,
                    false,
                ));
            }
            segments.extend(visual_line_segments(line, visual.frame + offset));
            self.put_segments_line(
                rows,
                body_start + top_padding as u16 + offset as u16,
                &segments,
            );
        }
    }

    fn render_prompt(&self, rows: &mut [RenderedRow], prompt: &PromptState) {
        let body_start = 5;
        let mut line = body_start;
        self.put_labeled_line(rows, line, "RUN", &prompt.message, LineKind::Run);
        line += 1;

        let available = self.height.saturating_sub(line + 3) as usize;
        let start = if prompt.selected >= available && available > 0 {
            prompt.selected + 1 - available
        } else {
            0
        };
        let end = (start + available).min(prompt.options.len());

        for index in start..end {
            let marker = if index == prompt.selected { ">" } else { " " };
            let option = format!("{marker} {}", &prompt.options[index]);
            let kind = if index == prompt.selected {
                LineKind::Ok
            } else {
                LineKind::Info
            };
            self.put_labeled_line(rows, line, "INFO", &option, kind);
            line += 1;
        }

        self.put_line(
            rows,
            self.height.saturating_sub(3),
            "Use arrows or j/k, Enter to select, Esc to cancel",
            Color::DarkGrey,
            false,
        );
    }

    fn render_footer(&self, rows: &mut [RenderedRow]) {
        self.put_line(
            rows,
            self.height.saturating_sub(2),
            &"-".repeat(self.width as usize),
            Color::DarkGrey,
            false,
        );
        self.put_line(
            rows,
            self.height.saturating_sub(1),
            &self.footer_text(),
            Color::DarkGrey,
            false,
        );
    }

    fn footer_text(&self) -> String {
        let base = "--plain for classic output | --verbose for raw subprocess output";
        if let Some(message) = self.footer_message.as_deref() {
            return format!("{message} | {base}");
        }
        if let Some(log_path) = self.current_log_path.as_ref() {
            return format!("Log: {} | {base}", display_path(log_path));
        }
        format!("Bounded Inline active | {base}")
    }

    fn put_labeled_line(
        &self,
        rows: &mut [RenderedRow],
        row: u16,
        label: &str,
        text: &str,
        kind: LineKind,
    ) {
        let Some(slot) = rows.get_mut(row as usize) else {
            return;
        };
        let label = format!("{label:<8}");
        let max_text = self.width.saturating_sub(label.len() as u16) as usize;
        let text = truncate_text(text, max_text);
        *slot = RenderedRow::segments([
            RowSegment::new(label, kind.color(), true),
            RowSegment::new(text, Color::White, false),
        ]);
    }

    fn put_styled_labeled_line(
        &self,
        rows: &mut [RenderedRow],
        row: u16,
        label: &str,
        segments: &[RowSegment],
        kind: LineKind,
    ) {
        let Some(slot) = rows.get_mut(row as usize) else {
            return;
        };
        let label = format!("{label:<8}");
        let max_text = self.width.saturating_sub(label.len() as u16) as usize;
        let mut row_segments = Vec::new();
        row_segments.push(RowSegment::new(label, kind.color(), true));
        row_segments.extend(truncate_segments(segments, max_text));
        *slot = RenderedRow {
            segments: row_segments,
        };
    }

    fn put_line(&self, rows: &mut [RenderedRow], row: u16, text: &str, color: Color, bold: bool) {
        let Some(slot) = rows.get_mut(row as usize) else {
            return;
        };
        let text = truncate_text(text, self.width as usize);
        *slot = RenderedRow::segments([RowSegment::new(text, color, bold)]);
    }

    fn put_segments_line(&self, rows: &mut [RenderedRow], row: u16, segments: &[RowSegment]) {
        let Some(slot) = rows.get_mut(row as usize) else {
            return;
        };
        *slot = RenderedRow {
            segments: truncate_segments(segments, self.width as usize),
        };
    }

    fn draw_rendered_row(
        &self,
        stdout: &mut std::io::Stdout,
        row: u16,
        rendered: &RenderedRow,
        clear_first: bool,
    ) {
        let y = self.origin_y.saturating_add(row);
        let _ = queue!(stdout, MoveTo(0, y));
        if clear_first {
            let _ = queue!(stdout, Clear(ClearType::CurrentLine));
        }

        let mut visible_width = 0usize;
        for segment in &rendered.segments {
            visible_width += segment.visible_width();
            let _ = queue!(
                stdout,
                SetAttribute(Attribute::Reset),
                SetForegroundColor(segment.color)
            );
            if segment.bold {
                let _ = queue!(stdout, SetAttribute(Attribute::Bold));
            }
            if segment.dim {
                let _ = queue!(stdout, SetAttribute(Attribute::Dim));
            }
            let _ = queue!(stdout, Print(&segment.text));
        }

        let width = self.width as usize;
        if visible_width < width {
            let _ = queue!(
                stdout,
                SetAttribute(Attribute::Reset),
                ResetColor,
                Print(" ".repeat(width - visible_width))
            );
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct RenderedRow {
    segments: Vec<RowSegment>,
}

impl RenderedRow {
    fn blank() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    fn segments<const N: usize>(segments: [RowSegment; N]) -> Self {
        Self {
            segments: Vec::from(segments),
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct RowSegment {
    text: String,
    color: Color,
    bold: bool,
    dim: bool,
}

impl RowSegment {
    fn new(text: impl Into<String>, color: Color, bold: bool) -> Self {
        Self {
            text: text.into(),
            color,
            bold,
            dim: false,
        }
    }

    fn styled(text: impl Into<String>, color: Color, bold: bool, dim: bool) -> Self {
        Self {
            text: text.into(),
            color,
            bold,
            dim,
        }
    }

    fn visible_width(&self) -> usize {
        measure_text_width(&self.text)
    }
}

fn truncate_segments(segments: &[RowSegment], width: usize) -> Vec<RowSegment> {
    if width == 0 {
        return Vec::new();
    }

    let mut remaining = width;
    let mut truncated = Vec::new();
    for segment in segments {
        if remaining == 0 {
            break;
        }

        let segment_width = segment.visible_width();
        if segment_width <= remaining {
            truncated.push(segment.clone());
            remaining -= segment_width;
            continue;
        }

        if remaining <= 3 {
            truncated.push(RowSegment::styled(
                ".".repeat(remaining),
                segment.color,
                segment.bold,
                segment.dim,
            ));
        } else {
            truncated.push(RowSegment::styled(
                truncate_text(&segment.text, remaining),
                segment.color,
                segment.bold,
                segment.dim,
            ));
        }
        break;
    }

    truncated
}

fn parse_ansi_segments(text: &str) -> Vec<RowSegment> {
    let mut parser = AnsiStyle::default();
    let mut segments = Vec::new();
    let mut buffer = String::new();
    let mut chars = text.chars().peekable();

    while let Some(character) = chars.next() {
        if character == '\u{1b}' {
            flush_ansi_buffer(&mut segments, &mut buffer, parser);
            if matches!(chars.peek(), Some('[')) {
                let _ = chars.next();
                let mut sequence = String::new();
                for next in chars.by_ref() {
                    if ('@'..='~').contains(&next) {
                        if next == 'm' {
                            parser.apply_sgr(&sequence);
                        }
                        break;
                    }
                    sequence.push(next);
                }
            } else if matches!(chars.peek(), Some(']')) {
                let _ = chars.next();
                while let Some(next) = chars.next() {
                    if next == '\u{7}' {
                        break;
                    }
                    if next == '\u{1b}' && matches!(chars.peek(), Some('\\')) {
                        let _ = chars.next();
                        break;
                    }
                }
            } else {
                let _ = chars.next();
            }
        } else if character == '\t' {
            buffer.push(' ');
        } else if !character.is_control() {
            buffer.push(character);
        }
    }

    flush_ansi_buffer(&mut segments, &mut buffer, parser);
    if segments.is_empty() {
        segments.push(RowSegment::new(String::new(), Color::White, false));
    }
    segments
}

fn flush_ansi_buffer(segments: &mut Vec<RowSegment>, buffer: &mut String, style: AnsiStyle) {
    if buffer.is_empty() {
        return;
    }

    segments.push(RowSegment::styled(
        std::mem::take(buffer),
        style.color,
        style.bold,
        style.dim,
    ));
}

#[derive(Clone, Copy)]
struct AnsiStyle {
    color: Color,
    bold: bool,
    dim: bool,
}

impl Default for AnsiStyle {
    fn default() -> Self {
        Self {
            color: Color::White,
            bold: false,
            dim: false,
        }
    }
}

impl AnsiStyle {
    fn apply_sgr(&mut self, sequence: &str) {
        let mut codes = sequence
            .split(';')
            .filter_map(|part| {
                if part.is_empty() {
                    Some(0)
                } else {
                    part.parse::<u16>().ok()
                }
            })
            .peekable();

        if codes.peek().is_none() {
            *self = Self::default();
            return;
        }

        while let Some(code) = codes.next() {
            match code {
                0 => *self = Self::default(),
                1 => self.bold = true,
                2 => self.dim = true,
                22 => {
                    self.bold = false;
                    self.dim = false;
                }
                38 => {
                    self.apply_extended_foreground(&mut codes);
                }
                30..=37 | 90..=97 => {
                    if let Some(color) = ansi_color(code) {
                        self.color = color;
                    }
                }
                39 => self.color = Color::White,
                _ => {}
            }
        }
    }

    fn apply_extended_foreground<I>(&mut self, codes: &mut I)
    where
        I: Iterator<Item = u16>,
    {
        let Some(mode) = codes.next() else {
            return;
        };

        match mode {
            2 => {
                let (Some(r), Some(g), Some(b)) = (codes.next(), codes.next(), codes.next()) else {
                    return;
                };
                let (Ok(r), Ok(g), Ok(b)) = (u8::try_from(r), u8::try_from(g), u8::try_from(b))
                else {
                    return;
                };
                self.color = Color::Rgb { r, g, b };
            }
            5 => {
                let Some(value) = codes.next() else {
                    return;
                };
                let Ok(value) = u8::try_from(value) else {
                    return;
                };
                self.color = Color::AnsiValue(value);
            }
            _ => {}
        }
    }
}

fn ansi_color(code: u16) -> Option<Color> {
    Some(match code {
        30 => Color::Black,
        31 => Color::DarkRed,
        32 => Color::DarkGreen,
        33 => Color::DarkYellow,
        34 => Color::DarkBlue,
        35 => Color::DarkMagenta,
        36 => Color::DarkCyan,
        37 => Color::Grey,
        90 => Color::DarkGrey,
        91 => Color::Red,
        92 => Color::Green,
        93 => Color::Yellow,
        94 => Color::Blue,
        95 => Color::Magenta,
        96 => Color::Cyan,
        97 => Color::White,
        _ => return None,
    })
}

#[derive(Clone, Copy)]
enum LineKind {
    Info,
    Path,
    Run,
    Output,
    Ok,
    Warn,
    Err,
    Artifact,
}

impl LineKind {
    fn color(self) -> Color {
        match self {
            LineKind::Info => Color::DarkGrey,
            LineKind::Path => Color::Cyan,
            LineKind::Run => Color::Blue,
            LineKind::Output => Color::White,
            LineKind::Ok => Color::Green,
            LineKind::Warn => Color::Yellow,
            LineKind::Err => Color::Red,
            LineKind::Artifact => Color::Magenta,
        }
    }
}

struct ActivityLine {
    kind: LineKind,
    label: &'static str,
    text: String,
}

struct OutputLine {
    kind: LineKind,
    label: &'static str,
    segments: Vec<RowSegment>,
}

struct FinalStatus {
    success: bool,
    summary: String,
}

struct ActiveStep {
    label: String,
    started_at: Instant,
    frame: usize,
}

struct InlineVisual {
    lines: Vec<String>,
    frame: usize,
}

#[derive(Clone)]
struct PromptState {
    message: String,
    options: Vec<String>,
    selected: usize,
}

fn spinner_frame(frame: usize) -> &'static str {
    match frame % 6 {
        0 => ".  ",
        1 => ".. ",
        2 => "...",
        3 => " ..",
        4 => "  .",
        _ => "   ",
    }
}

fn active_step_message(label: &str, duration: Duration) -> String {
    format!("{label} ({})", elapsed(duration))
}

fn visual_color(frame: usize) -> Color {
    match frame % 6 {
        0 => Color::Cyan,
        1 => Color::Blue,
        2 => Color::Magenta,
        3 => Color::White,
        4 => Color::DarkCyan,
        _ => Color::DarkBlue,
    }
}

fn visual_line_segments(text: &str, frame: usize) -> Vec<RowSegment> {
    if text.contains('\u{1b}') {
        parse_ansi_segments(text)
    } else {
        vec![RowSegment::new(text, visual_color(frame), true)]
    }
}

fn truncate_text(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    if measure_text_width(text) <= width {
        return text.to_owned();
    }
    if width <= 3 {
        return ".".repeat(width);
    }

    truncate_str(text, width, "...").into_owned()
}
