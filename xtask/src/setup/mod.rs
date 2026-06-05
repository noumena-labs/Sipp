//! Interactive setup guide for the CogentLM workspace.

mod launcher;

use crate::cli::{SetupArgs, SetupProfile};
use crate::javascript;
use crate::output;
use crate::sample_model::{self, SampleModelOptions};
use crate::terminal::splash;
use crate::toolchain;
use crate::toolchains::{emsdk, ninja, python, vulkan};
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use console::Term;
use inquire::{Confirm, Select};
use std::env;
use std::fmt;
use std::path::PathBuf;
use xshell::Shell;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/setup_tests.rs"]
mod setup_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Runs the guided local setup flow.
pub fn run(sh: &Shell, ctx: &BuildContext, args: &SetupArgs) -> Result<()> {
    let splash_played = splash::play(args.no_splash)?;
    if !splash_played {
        output::banner("COGENTLM");
    }

    output::phase("CogentLM setup");
    output::path("Workspace", ctx.workspace_root());
    output::detail(
        "Downloads",
        if args.no_downloads {
            "disabled"
        } else {
            "ask first"
        },
    );

    let interactive = can_prompt();
    let profile = select_profile(args, interactive)?;
    output::detail("Profile", profile.as_str());

    print_readiness(ctx);

    if should_install_launcher(args, interactive)? {
        output::phase("Install clm launcher");
        launcher::install(sh, ctx)?;
    } else {
        output::detail("Launcher", "skipped");
    }

    let downloads = select_downloads(profile, args, interactive)?;
    run_downloads(sh, ctx, profile, &downloads)?;
    print_examples(ctx, profile);
    output::success("Setup guide complete");
    Ok(())
}

fn can_prompt() -> bool {
    Term::stdout().is_term() && env::var_os("CI").is_none()
}

fn select_profile(args: &SetupArgs, interactive: bool) -> Result<SetupProfile> {
    if let Some(profile) = args.profile {
        return Ok(profile);
    }

    if !interactive {
        return Ok(SetupProfile::Browser);
    }

    if output::inline_active() {
        let profiles = [
            SetupProfile::Browser,
            SetupProfile::Bindings,
            SetupProfile::Full,
        ];
        let options = profiles.iter().map(ToString::to_string).collect::<Vec<_>>();
        if let Some(index) =
            output::prompt_select("What do you want to set up first?", &options, 0)?
        {
            return Ok(profiles[index]);
        }
    }

    Select::new(
        "What do you want to set up first?",
        vec![
            SetupProfile::Browser,
            SetupProfile::Bindings,
            SetupProfile::Full,
        ],
    )
    .with_help_message("This only changes which setup steps and example commands are recommended.")
    .prompt()
    .context("failed to read setup profile selection")
}

fn print_readiness(ctx: &BuildContext) {
    output::phase("Readiness snapshot");
    for status in toolchain::managed_statuses(ctx) {
        status.print();
    }
    for status in toolchain::external_statuses(ctx) {
        status.print();
    }
}

fn should_install_launcher(args: &SetupArgs, interactive: bool) -> Result<bool> {
    if args.yes {
        return Ok(true);
    }
    if !interactive {
        return Ok(true);
    }

    if let Some(answer) = output::prompt_confirm(
        "Install the repo-local clm launcher under .build/bin?",
        true,
    )? {
        return Ok(answer);
    }

    Confirm::new("Install the repo-local clm launcher under .build/bin?")
        .with_default(true)
        .prompt()
        .context("failed to read launcher setup confirmation")
}

fn select_downloads(
    profile: SetupProfile,
    args: &SetupArgs,
    interactive: bool,
) -> Result<Vec<SetupDownload>> {
    if args.no_downloads {
        return Ok(Vec::new());
    }

    let recommended = recommended_downloads(profile);
    if args.yes {
        return Ok(recommended);
    }

    let mut selected = Vec::new();
    for download in recommended {
        if should_run_download(download, interactive)? {
            selected.push(download);
        }
    }
    Ok(selected)
}

fn should_run_download(download: SetupDownload, interactive: bool) -> Result<bool> {
    if !interactive {
        return Ok(false);
    }

    let prompt = format!("{download}?");
    if let Some(answer) = output::prompt_confirm(&prompt, false)? {
        return Ok(answer);
    }

    Confirm::new(&prompt)
        .with_default(false)
        .prompt()
        .with_context(|| format!("failed to read confirmation for {download}"))
}

fn recommended_downloads(profile: SetupProfile) -> Vec<SetupDownload> {
    match profile {
        SetupProfile::Browser => vec![
            SetupDownload::ManagedToolchains,
            SetupDownload::JavaScriptDependencies,
        ],
        SetupProfile::Bindings => vec![
            SetupDownload::ManagedToolchains,
            SetupDownload::JavaScriptDependencies,
            SetupDownload::SampleModel,
        ],
        SetupProfile::Full => vec![
            SetupDownload::ManagedToolchains,
            SetupDownload::JavaScriptDependencies,
            SetupDownload::SampleModel,
        ],
    }
}

fn run_downloads(
    sh: &Shell,
    ctx: &BuildContext,
    profile: SetupProfile,
    downloads: &[SetupDownload],
) -> Result<()> {
    if downloads.is_empty() {
        output::phase("Downloads skipped");
        output::detail("Toolchains", "cargo xtask toolchain install all");
        output::detail("JavaScript", "cargo xtask setup --profile browser --yes");
        output::detail("Sample model", sample_model::sample_model_url());
        return Ok(());
    }

    if downloads.contains(&SetupDownload::ManagedToolchains) {
        install_managed_toolchains(sh, ctx, profile)?;
    }
    if downloads.contains(&SetupDownload::JavaScriptDependencies) {
        install_javascript_dependencies(sh, ctx, profile)?;
    }
    if downloads.contains(&SetupDownload::SampleModel) {
        download_sample_model(sh, ctx)?;
    }

    Ok(())
}

fn install_managed_toolchains(sh: &Shell, ctx: &BuildContext, profile: SetupProfile) -> Result<()> {
    output::phase("Managed toolchains");
    match profile {
        SetupProfile::Browser => {
            ninja::setup_ninja(sh, ctx)?;
            emsdk::setup_emsdk(sh, ctx)?;
        }
        SetupProfile::Bindings => {
            python::setup_uv(sh, ctx)?;
            vulkan::setup_vulkan(sh, ctx)?;
        }
        SetupProfile::Full => {
            python::setup_uv(sh, ctx)?;
            ninja::setup_ninja(sh, ctx)?;
            emsdk::setup_emsdk(sh, ctx)?;
            vulkan::setup_vulkan(sh, ctx)?;
        }
    }
    Ok(())
}

fn install_javascript_dependencies(
    sh: &Shell,
    ctx: &BuildContext,
    profile: SetupProfile,
) -> Result<()> {
    output::phase("JavaScript dependencies");

    let package_dirs = root_javascript_package_dirs(ctx, profile)?;
    if !package_dirs.is_empty() {
        javascript::install_root_workspace_dependencies(
            sh,
            ctx,
            "Installing JavaScript workspace dependencies",
            &package_dirs,
        )?;
    }

    if should_install_node_binding_javascript_dependencies(profile) {
        javascript::install_node_binding_dependencies(sh, ctx)?;
    }

    Ok(())
}

fn root_javascript_package_dirs(ctx: &BuildContext, profile: SetupProfile) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    if matches!(profile, SetupProfile::Browser | SetupProfile::Full) {
        dirs.extend(browser_javascript_package_dirs(ctx)?);
    }
    if matches!(profile, SetupProfile::Bindings | SetupProfile::Full) {
        dirs.push(ctx.node_package_dir());
    }
    Ok(dirs)
}

fn browser_javascript_package_dirs(ctx: &BuildContext) -> Result<Vec<PathBuf>> {
    let mut dirs = vec![ctx.browser_package_dir(), ctx.browser_example_dir()];
    dirs.extend(ctx.demo_dirs()?);
    dirs.extend(ctx.benchmark_dirs()?);
    Ok(dirs)
}

fn should_install_node_binding_javascript_dependencies(profile: SetupProfile) -> bool {
    matches!(profile, SetupProfile::Bindings | SetupProfile::Full)
}

fn download_sample_model(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    sample_model::ensure_sample_model(
        sh,
        ctx,
        SampleModelOptions {
            allow_download: true,
        },
    )
    .map(|_| ())
}

fn print_examples(ctx: &BuildContext, profile: SetupProfile) {
    output::phase("Next commands");
    output::detail("Doctor", "clm doctor");
    match profile {
        SetupProfile::Browser => {
            output::detail("Build browser package", "clm build wasm");
            output::detail("Run chat demo", "clm run demos serve chat");
            output::detail("Run demo tests", "clm test unit suite demos");
        }
        SetupProfile::Bindings => {
            let model = sample_model::sample_model_arg(ctx);
            output::detail("Build Node bindings", "clm build node");
            output::detail("Build Python bindings", "clm build python");
            output::detail(
                "Node API tests",
                "clm test unit suite node-package --backend cpu",
            );
            output::detail(
                "Python API tests",
                "clm test unit suite python-package --backend cpu",
            );
            output::detail(
                "Model smoke",
                format!("clm test smoke group local-model --backend cpu --model {model}"),
            );
        }
        SetupProfile::Full => {
            let model = sample_model::sample_model_arg(ctx);
            output::detail("Build everything", "clm build all");
            output::detail("Serve avatar demo", "clm run demos serve avatar");
            output::detail("Run unit tests", "clm test unit group full");
            output::detail(
                "Run smoke tests",
                format!("clm test smoke group full --backend cpu --model {model}"),
            );
        }
    }
    output::detail("Compatibility", "cargo xtask still works for every command");
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SetupDownload {
    ManagedToolchains,
    JavaScriptDependencies,
    SampleModel,
}

impl fmt::Display for SetupDownload {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            SetupDownload::ManagedToolchains => "Download or activate missing managed toolchains",
            SetupDownload::JavaScriptDependencies => "Install JavaScript workspace dependencies",
            SetupDownload::SampleModel => "Download the small Qwen sample GGUF model",
        })
    }
}
