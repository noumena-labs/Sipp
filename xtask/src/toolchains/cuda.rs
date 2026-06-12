//! CUDA toolkit discovery and architecture configuration.

use crate::output;
use crate::utils::BuildContext;
use anyhow::Result;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/toolchains/cuda_tests.rs"]
mod cuda_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Portable cloud GPU architectures for CUDA 13 builds.
///
/// Covers T4 (75), A100 (80), A10/A40-class Ampere (86), L4/L40S Ada (89),
/// H100/H200 Hopper (90), and the Blackwell architecture-specific targets
/// used by vendored llama.cpp (120a, 121a).
pub(crate) const DEFAULT_CUDA_ARCHITECTURES: &str =
    "75-virtual;80-virtual;86-real;89-real;90-virtual;120a-real;121a-real";

/// Where a resolved architecture list comes from — used for status reporting.
#[derive(Clone, Debug)]
pub(crate) enum ArchSource {
    EnvVar,
    ConfigFile(PathBuf),
    Default,
}

impl ArchSource {
    pub(crate) fn label(&self) -> &'static str {
        match self {
            ArchSource::EnvVar => "env var",
            ArchSource::ConfigFile(_) => "config file",
            ArchSource::Default => "default",
        }
    }

    pub(crate) fn config_path(&self) -> Option<&Path> {
        match self {
            ArchSource::ConfigFile(p) => Some(p.as_path()),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Config file persistence under .build/config/cuda-architectures
// ---------------------------------------------------------------------------

pub(crate) fn cuda_config_path(ctx: &BuildContext) -> PathBuf {
    ctx.config_dir().join("cuda-architectures")
}

pub(crate) fn read_persisted_architectures(ctx: &BuildContext) -> Option<String> {
    let path = cuda_config_path(ctx);
    if !path.exists() {
        return None;
    }
    let content = fs::read_to_string(path).ok()?;
    let trimmed = content.trim().to_owned();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

pub(crate) fn write_persisted_architectures(ctx: &BuildContext, arches: &str) -> Result<()> {
    let path = cuda_config_path(ctx);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, arches)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Architecture list resolution (priority: env var > config file > default)
// ---------------------------------------------------------------------------

/// Resolves the CUDA architecture list together with its source for status
/// reporting.  Honors `SIPP_CUDA_ARCHITECTURES` (env var, highest
/// priority), then the persisted config file written by `toolchain setup
/// cuda`, and finally the built-in default.
pub(crate) fn cuda_architectures_with_source(ctx: &BuildContext) -> (String, ArchSource) {
    let from_env = env::var("SIPP_CUDA_ARCHITECTURES")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());

    if let Some(arches) = from_env {
        return (arches, ArchSource::EnvVar);
    }

    if let Some(arches) = read_persisted_architectures(ctx) {
        return (arches, ArchSource::ConfigFile(cuda_config_path(ctx)));
    }

    (DEFAULT_CUDA_ARCHITECTURES.to_owned(), ArchSource::Default)
}

/// Resolves the CUDA architecture list for xtask-driven CUDA builds.
///
/// Checks `SIPP_CUDA_ARCHITECTURES` (env var), then the persisted config
/// file written by `toolchain setup cuda`, and finally the portable cloud
/// default.
pub(crate) fn cuda_architectures(ctx: &BuildContext) -> String {
    cuda_architectures_with_source(ctx).0
}

// ---------------------------------------------------------------------------
// GPU detection
// ---------------------------------------------------------------------------

/// Detects the compute capability of the first NVIDIA GPU via `nvidia-smi`.
///
/// Returns something like `"89-real"` for a CC 8.9 device, or `None` when
/// `nvidia-smi` is unavailable or no GPU is found.
pub(crate) fn detect_device_arch() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let first_line = stdout.lines().next()?;
    let cap = first_line.trim();
    if cap.is_empty() {
        return None;
    }
    compute_cap_to_arch(cap)
}

pub(crate) fn compute_cap_to_arch(cap: &str) -> Option<String> {
    let compact: String = cap.chars().filter(|c| c.is_ascii_digit()).collect();
    if compact.is_empty() {
        return None;
    }
    Some(format!("{compact}-real"))
}

// ---------------------------------------------------------------------------
// nvcc query helpers
// ---------------------------------------------------------------------------

fn nvcc_path(cuda_root: &Path) -> PathBuf {
    let exe = if cfg!(windows) { "nvcc.exe" } else { "nvcc" };
    cuda_root.join("bin").join(exe)
}

/// Returns architecture names that the installed nvcc supports, with the
/// `sm_` prefix stripped (e.g. `"75"`, `"90a"`, `"120"`).
pub(crate) fn nvcc_supported_arches(nvcc: &Path) -> Vec<String> {
    let output = Command::new(nvcc).arg("--list-gpu-arch").output().ok();

    let Some(output) = output else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.strip_prefix("sm_").unwrap_or(l).to_owned())
        .collect()
}

// ---------------------------------------------------------------------------
// CUDA toolkit discovery
// ---------------------------------------------------------------------------

/// Finds a CUDA toolkit installation suitable for CMake/NVCC builds.
pub(crate) fn setup_cuda() -> Result<PathBuf> {
    output::phase("CUDA Toolkit");

    let cuda_env = env::var_os("CUDA_PATH").or_else(|| env::var_os("CUDA_HOME"));

    if let Some(path) = cuda_env {
        let cuda_path = PathBuf::from(path);
        let nvcc_exe = if cfg!(windows) {
            cuda_path.join("bin").join("nvcc.exe")
        } else {
            cuda_path.join("bin").join("nvcc")
        };

        if nvcc_exe.exists() {
            output::success(format!("Using CUDA Toolkit at {}", cuda_path.display()));
            return Ok(cuda_path);
        }
    }

    output::warning("CUDA Toolkit not found");
    output::detail("Required for", "CUDA backend builds");
    output::detail("Download", "https://developer.nvidia.com/cuda-downloads");
    output::detail("Expected env", "CUDA_PATH or CUDA_HOME pointing at nvcc");
    output::detail("Retry", "cargo xtask build node --backend cuda");
    anyhow::bail!("Missing NVIDIA CUDA Toolkit.");
}

// ---------------------------------------------------------------------------
// Interactive setup
// ---------------------------------------------------------------------------

/// Runs the interactive CUDA architecture setup.
///
/// Detects the physical GPU, shows the current configuration, presents
/// presets and custom options, and persists the choice so subsequent builds
/// use the selected architecture list.
pub(crate) fn setup_cuda_architectures(ctx: &BuildContext) -> Result<()> {
    output::phase("CUDA architecture setup");

    let cuda_root = setup_cuda()?;
    let nvcc = nvcc_path(&cuda_root);

    // Detect physical GPU
    let detected = detect_device_arch();
    if let Some(ref arch) = detected {
        output::success(format!("Detected GPU compute capability: {arch}"));
    } else {
        output::warning("Could not detect GPU via nvidia-smi");
        output::detail(
            "Tip",
            "Install NVIDIA drivers or check nvidia-smi availability",
        );
    }

    // Show current configuration
    let (current, source) = cuda_architectures_with_source(ctx);
    output::detail(
        "Current architecture list",
        format!("[{}] {current}", source.label()),
    );

    let interactive = output::inline_active();
    let selected = if interactive {
        select_arches_inline(&detected, &current)?
    } else {
        select_arches_interactive(&detected, &current, &nvcc)?
    };

    if selected == current {
        output::detail("Architecture list", "unchanged");
        return Ok(());
    }

    write_persisted_architectures(ctx, &selected)?;
    output::success("CUDA architecture list saved");
    output::path("Config file", &cuda_config_path(ctx));

    let shell_hint = if cfg!(windows) {
        format!("$env:SIPP_CUDA_ARCHITECTURES = \"{selected}\"")
    } else {
        format!("export SIPP_CUDA_ARCHITECTURES=\"{selected}\"")
    };
    output::detail("Override per session", shell_hint);

    Ok(())
}

// ---------------------------------------------------------------------------
// Preset helpers
// ---------------------------------------------------------------------------

fn suggested_presets(detected: &Option<String>, current: &str) -> Vec<(String, String)> {
    let mut presets: Vec<(String, String)> = Vec::new();

    if let Some(ref arch) = detected {
        presets.push((format!("Only my GPU ({arch})"), arch.clone()));
        presets.push((
            format!("My GPU + portable cloud ({arch};80-virtual;90-virtual)"),
            format!("{arch};80-virtual;90-virtual"),
        ));
    }

    presets.push((
        "Portable cloud default (recommended for shared artifacts)".to_owned(),
        DEFAULT_CUDA_ARCHITECTURES.to_owned(),
    ));

    if current != DEFAULT_CUDA_ARCHITECTURES
        && detected.as_ref().is_none_or(|d| current != d.as_str())
    {
        presets.push(("Restore previous selection".to_owned(), current.to_owned()));
    }

    presets.push((
        "Custom (enter architecture string)".to_owned(),
        String::new(),
    ));
    presets.push((
        "Full nvcc-supported list (multi-select)".to_owned(),
        String::new(),
    ));

    presets
}

fn compute_cap_sort_key(arch: &str) -> u32 {
    let num_part: String = arch.chars().take_while(|c| c.is_ascii_digit()).collect();
    num_part.parse::<u32>().unwrap_or(u32::MAX)
}

// ---------------------------------------------------------------------------
// Interactive selector (inquire-based, used when terminal is available)
// ---------------------------------------------------------------------------

fn select_arches_interactive(
    detected: &Option<String>,
    current: &str,
    nvcc: &Path,
) -> Result<String> {
    let presets = suggested_presets(detected, current);

    let preset_labels: Vec<String> = presets.iter().map(|(label, _)| label.clone()).collect();

    let selection = inquire::Select::new(
        "Select CUDA architecture targets for the build",
        preset_labels,
    )
    .with_starting_cursor(0)
    .with_help_message("↑↓ navigate, Enter to select")
    .prompt()?;

    let index = presets
        .iter()
        .position(|(label, _)| label == &selection)
        .unwrap_or(0);

    if index < presets.len() {
        let (_, value) = &presets[index];
        if value.is_empty() && selection.starts_with("Custom") {
            return prompt_custom_arch();
        }
        if value.is_empty() && selection.starts_with("Full") {
            return prompt_multi_select_from_nvcc(nvcc);
        }
        return Ok(value.clone());
    }

    Ok(current.to_owned())
}

fn prompt_custom_arch() -> Result<String> {
    let custom = inquire::Text::new("Enter CUDA architecture list (semicolon-separated)")
        .with_help_message("e.g. 89-real or 80-virtual;89-real;90-virtual")
        .prompt()?;
    let trimmed = custom.trim().to_owned();
    if trimmed.is_empty() {
        anyhow::bail!("Architecture string cannot be empty");
    }
    Ok(trimmed)
}

fn prompt_multi_select_from_nvcc(nvcc: &Path) -> Result<String> {
    let all_arches = nvcc_supported_arches(nvcc);
    if all_arches.is_empty() {
        output::warning("Could not list architectures from nvcc");
        return prompt_custom_arch();
    }

    let mut sorted = all_arches.clone();
    sorted.sort_by_key(|a| compute_cap_sort_key(a));

    let mut options: Vec<String> = sorted.iter().map(|a| format!("{a}-real")).collect();
    // Also offer virtual variants for common ones
    let virtual_options: Vec<String> = sorted.iter().map(|a| format!("{a}-virtual")).collect();
    options.extend(virtual_options);

    let selections = inquire::MultiSelect::new(
        "Select CUDA architectures to compile for (Space to toggle, Enter to confirm)",
        options,
    )
    .with_help_message("Space to select, ↑↓ to navigate, Enter when done")
    .prompt()?;

    if selections.is_empty() {
        anyhow::bail!("At least one architecture must be selected");
    }

    Ok(selections.join(";"))
}

// ---------------------------------------------------------------------------
// Inline selector (used when bounded inline rendering is active)
// ---------------------------------------------------------------------------

fn select_arches_inline(detected: &Option<String>, current: &str) -> Result<String> {
    let presets = suggested_presets(detected, current);
    let preset_labels: Vec<String> = presets.iter().map(|(label, _)| label.clone()).collect();

    let Some(index) = output::prompt_select(
        "Select CUDA architecture targets for the build",
        &preset_labels,
        0,
    )?
    else {
        return Ok(current.to_owned());
    };

    let (_, value) = &presets[index];
    if value.is_empty() && preset_labels[index].starts_with("Custom") {
        output::detail(
            "Custom entry",
            "use `cargo xtask toolchain setup cuda` in a terminal",
        );
        return Ok(current.to_owned());
    }
    if value.is_empty() && preset_labels[index].starts_with("Full") {
        output::detail(
            "Full list",
            "use `cargo xtask toolchain setup cuda` in a terminal",
        );
        return Ok(current.to_owned());
    }

    Ok(value.clone())
}
