//! Terminal output helpers for xtask.

use crate::utils::BuildContext;
use anyhow::{anyhow, Context, Result};
use console::{style, Term};
use crossterm::cursor::{position, Hide, MoveTo, MoveToPreviousLine, Show};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{Attribute, Color, Print, ResetColor, SetAttribute, SetForegroundColor};
use crossterm::terminal::{self, Clear, ClearType, DisableLineWrap, EnableLineWrap};
use crossterm::{execute, queue};
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::VecDeque;
use std::env;
use std::fmt::Display;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write as _};
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tui_banner::{Align, Banner, Style as BannerStyle};
use xshell::Cmd;

const COMMAND_TAIL_LINES: usize = 20;
const INLINE_DEFAULT_LINES: u16 = 18;
const INLINE_MIN_LINES: u16 = 12;
const INLINE_MIN_COLUMNS: u16 = 60;
const INLINE_MIN_RENDER_INTERVAL: Duration = Duration::from_millis(50);
const MAX_ACTIVITY_LINES: usize = 160;

static VERBOSE: AtomicBool = AtomicBool::new(false);
static NO_BANNER: AtomicBool = AtomicBool::new(false);
static COMMAND_LOG_COUNTER: AtomicUsize = AtomicUsize::new(0);
static COMMAND_LOG_DIR: OnceLock<PathBuf> = OnceLock::new();
static INLINE_RENDERER: OnceLock<Mutex<Option<InlineRenderer>>> = OnceLock::new();

/// Configures terminal output for the current xtask invocation.
pub(crate) fn init(ctx: &BuildContext, verbose: bool, no_banner: bool, plain: bool) {
    VERBOSE.store(verbose, Ordering::Relaxed);
    NO_BANNER.store(no_banner, Ordering::Relaxed);
    let _ = COMMAND_LOG_DIR.set(ctx.command_logs_dir());

    let renderer = if should_use_inline(plain, verbose) {
        InlineRenderer::new(command_label()).ok().flatten()
    } else {
        None
    };

    if let Ok(mut slot) = inline_slot().lock() {
        *slot = renderer;
    }
}

/// Restores the terminal after bounded inline rendering.
pub(crate) fn finish() {
    let Some(renderer) = inline_slot().lock().ok().and_then(|mut slot| slot.take()) else {
        return;
    };
    renderer.finish();
}

/// Returns whether subprocess output should stream live.
pub(crate) fn verbose() -> bool {
    VERBOSE.load(Ordering::Relaxed)
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
        renderer.current_phase = Some(title.to_owned());
        renderer.add_event(LineKind::Run, "RUN", title);
        renderer.render_now();
    })
    .is_some()
    {
        return;
    }

    println!();
    println!("{} {}", style("==").cyan().bold(), style(title).bold().cyan());
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
    let message = format!("{label}: {}", path.display());
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
        style(path.display()).cyan()
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

    println!("   {} {}", style("OK").green().bold(), style(message).green());
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

    eprintln!("   {} {}", style("WARN").yellow().bold(), style(message).yellow());
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
    let label = label.into();
    if verbose() {
        return run_command_verbose(label, command);
    }

    if inline_active() {
        return run_command_inline(label, command);
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
    Err(anyhow!(
        "{} failed with {}",
        label,
        command_status_label(&output)
    ))
}

/// Runs a long-lived subprocess while keeping logs bounded in the inline viewport.
pub(crate) fn run_long_command(label: impl Into<String>, command: Cmd<'_>) -> Result<()> {
    let label = label.into();
    if verbose() || !inline_active() {
        return run_command_verbose(label, command);
    }

    let started_at = Instant::now();
    start_inline_step(&label);
    let (log_path, log_file) = open_command_log(&label)?;
    let log_file = Arc::new(Mutex::new(log_file));

    let mut process: std::process::Command = command.quiet().into();
    process
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = match process.spawn() {
        Ok(child) => child,
        Err(error) => {
            finish_inline_step(&label, started_at, false);
            return Err(error).with_context(|| format!("{label} failed to start"));
        }
    };

    let stdout_thread = child
        .stdout
        .take()
        .map(|stream| spawn_output_reader("stdout", stream, Arc::clone(&log_file)));
    let stderr_thread = child
        .stderr
        .take()
        .map(|stream| spawn_output_reader("stderr", stream, Arc::clone(&log_file)));
    let ticker = InlineTicker::start();

    let status = child
        .wait()
        .with_context(|| format!("{label} failed while waiting for process exit"))?;
    drop(ticker);
    join_output_reader(stdout_thread)?;
    join_output_reader(stderr_thread)?;

    if status.success() {
        finish_inline_step(&label, started_at, true);
        return Ok(());
    }

    finish_inline_step(&label, started_at, false);
    print_status_failure(&label, status, &log_path);
    Err(anyhow!(
        "{} failed with {}",
        label,
        exit_status_label(status)
    ))
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

fn run_command_inline(label: String, command: Cmd<'_>) -> Result<()> {
    let started_at = Instant::now();
    start_inline_step(&label);
    let ticker = InlineTicker::start();
    let output = command
        .quiet()
        .ignore_status()
        .output()
        .with_context(|| format!("{label} failed to start"))?;
    drop(ticker);

    if output.status.success() {
        finish_inline_step(&label, started_at, true);
        return Ok(());
    }

    finish_inline_step(&label, started_at, false);
    let log_path = write_command_log(&label, &output)?;
    print_command_failure(&label, &output, &log_path);
    Err(anyhow!(
        "{} failed with {}",
        label,
        command_status_label(&output)
    ))
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

        let lines = tail_output_lines(output, COMMAND_TAIL_LINES);
        if lines.is_empty() {
            detail("Recent output", "No subprocess output captured.");
            return;
        }
        detail("Recent output", "last subprocess lines");
        for line in lines {
            subprocess_event(&line, false);
        }
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
        style(log_path.display()).cyan()
    );

    let lines = tail_output_lines(output, COMMAND_TAIL_LINES);
    if lines.is_empty() {
        eprintln!("      {}", style("No subprocess output captured.").dim());
        return;
    }

    eprintln!("      {}", style("Recent subprocess output:").dim());
    for line in lines {
        eprintln!("      {}", style(line).dim());
    }
}

fn print_status_failure(label: &str, status: ExitStatus, log_path: &Path) {
    if inline_active() {
        error_event(format!("{label} failed with {}", exit_status_label(status)));
        path("Log", log_path);
        return;
    }

    eprintln!(
        "   {} {} failed with {}",
        style("ERR").red().bold(),
        label,
        exit_status_label(status)
    );
    eprintln!(
        "   {} {}",
        style("PATH").cyan().bold(),
        style(log_path.display()).cyan()
    );
}

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
    exit_status_label(output.status)
}

fn exit_status_label(status: ExitStatus) -> String {
    status
        .code()
        .map(|code| format!("exit code {code}"))
        .unwrap_or_else(|| "terminated by signal".to_owned())
}

fn spawn_output_reader<R>(
    stream_name: &'static str,
    stream: R,
    log_file: Arc<Mutex<File>>,
) -> JoinHandle<Result<()>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut reader = BufReader::new(stream);
        let mut bytes = Vec::new();

        loop {
            bytes.clear();
            let read = reader
                .read_until(b'\n', &mut bytes)
                .with_context(|| format!("failed to read {stream_name}"))?;
            if read == 0 {
                break;
            }

            if let Ok(mut log) = log_file.lock() {
                let _ = write!(log, "[{stream_name}] ");
                let _ = log.write_all(&bytes);
                if !bytes.ends_with(b"\n") {
                    let _ = writeln!(log);
                }
            }

            let line = String::from_utf8_lossy(&bytes);
            let line = line.trim_end_matches(&['\r', '\n'][..]);
            if !line.trim().is_empty() {
                subprocess_event(line, stream_name == "stderr");
            }
        }

        Ok(())
    })
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

fn start_inline_step(label: &str) {
    let _ = with_inline(|renderer| {
        renderer.active = Some(ActiveStep {
            label: label.to_owned(),
            started_at: Instant::now(),
            frame: 0,
        });
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
        } else {
            renderer.add_event(LineKind::Err, "ERR", &message);
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

fn subprocess_event(line: &str, stderr: bool) {
    let _ = with_inline(|renderer| {
        if stderr {
            renderer.add_event(LineKind::Warn, "WARN", line);
        } else {
            renderer.add_event(LineKind::Run, "RUN", line);
        }
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
                spinner.set_style(style.tick_strings(&[
                    ".  ", ".. ", "...", " ..", "  .", "   ",
                ]));
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
            visual: None,
            prompt: None,
            previous_rows,
            last_rendered_at: Instant::now(),
        };
        renderer.render_now();
        Ok(Some(renderer))
    }

    fn finish(mut self) {
        self.active = None;
        self.prompt = None;
        self.visual = None;
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
            let changed = self
                .previous_rows
                .get(row)
                .and_then(Option::as_ref)
                != Some(&rendered);
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
        let title = format!(" COGENTLM | {} | {}", self.command_label, elapsed);
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

        let active = self
            .active
            .as_ref()
            .map(|step| {
                format!(
                    "{} {} ({})",
                    spinner_frame(step.frame),
                    step.label,
                    elapsed(step.started_at.elapsed())
                )
            })
            .unwrap_or_else(|| "Idle".to_owned());
        self.put_labeled_line(rows, 3, "OK", &active, LineKind::Ok);
    }

    fn render_activity(&self, rows: &mut [RenderedRow]) {
        let body_start = 4;
        let body_end = self.height.saturating_sub(2);
        let body_lines = body_end.saturating_sub(body_start) as usize;
        if body_lines == 0 {
            return;
        }

        let start = self.activity.len().saturating_sub(body_lines);
        for (offset, line) in self.activity.iter().skip(start).enumerate() {
            self.put_labeled_line(
                rows,
                body_start + offset as u16,
                line.label,
                &line.text,
                line.kind,
            );
        }
    }

    fn render_visual(&self, rows: &mut [RenderedRow], visual: &InlineVisual) {
        let body_start = 4;
        let body_end = self.height.saturating_sub(2);
        let body_lines = body_end.saturating_sub(body_start) as usize;
        if body_lines == 0 {
            return;
        }

        let visible = visual.lines.iter().take(body_lines).collect::<Vec<_>>();
        let top_padding = body_lines.saturating_sub(visible.len()) / 2;
        for (offset, line) in visible.into_iter().enumerate() {
            let color = visual_color(visual.frame + offset);
            let centered = center_line(line, self.width as usize);
            self.put_line(
                rows,
                body_start + top_padding as u16 + offset as u16,
                &centered,
                color,
                true,
            );
        }
    }

    fn render_prompt(&self, rows: &mut [RenderedRow], prompt: &PromptState) {
        let body_start = 4;
        let mut line = body_start;
        self.put_labeled_line(rows, line, "RUN", &prompt.message, LineKind::Run);
        line += 1;

        let available = self.height.saturating_sub(line + 2) as usize;
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
            "Bounded Inline active | --plain for classic output | --verbose for live subprocess output",
            Color::DarkGrey,
            false,
        );
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

    fn put_line(
        &self,
        rows: &mut [RenderedRow],
        row: u16,
        text: &str,
        color: Color,
        bold: bool,
    ) {
        let Some(slot) = rows.get_mut(row as usize) else {
            return;
        };
        let text = truncate_text(text, self.width as usize);
        *slot = RenderedRow::segments([RowSegment::new(text, color, bold)]);
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
}

impl RowSegment {
    fn new(text: impl Into<String>, color: Color, bold: bool) -> Self {
        Self {
            text: text.into(),
            color,
            bold,
        }
    }

    fn visible_width(&self) -> usize {
        self.text.chars().count()
    }
}

#[derive(Clone, Copy)]
enum LineKind {
    Info,
    Path,
    Run,
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

fn center_line(text: &str, width: usize) -> String {
    let text_width = text.chars().count();
    if text_width >= width {
        return truncate_text(text, width);
    }

    let padding = (width - text_width) / 2;
    format!("{}{}", " ".repeat(padding), text)
}

fn truncate_text(text: &str, width: usize) -> String {
    if width == 0 {
        return String::new();
    }

    let count = text.chars().count();
    if count <= width {
        return text.to_owned();
    }
    if width <= 3 {
        return ".".repeat(width);
    }

    let mut truncated = text.chars().take(width - 3).collect::<String>();
    truncated.push_str("...");
    truncated
}
