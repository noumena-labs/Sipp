//! Interactive setup guide for the CogentLM workspace.

use crate::cli::{SetupArgs, SetupProfile};
use crate::launcher;
use crate::output;
use crate::splash;
use crate::toolchain;
use crate::toolchains::{emsdk, ninja, python, vulkan};
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use console::Term;
use inquire::{Confirm, Select};
use std::env;
use std::fmt;
use std::path::PathBuf;
use xshell::{cmd, Shell};

const SAMPLE_MODEL_URL: &str =
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf";
const SAMPLE_MODEL_FILE: &str = "qwen2.5-0.5b-instruct-q4_0.gguf";

/// Runs the guided local setup flow.
pub fn run(sh: &Shell, ctx: &BuildContext, args: &SetupArgs) -> Result<()> {
    let splash_played = splash::play(args.no_splash)?;
    if !splash_played {
        output::banner("COGENTLM");
    }

    output::phase("CogentLM setup");
    output::path("Workspace", ctx.workspace_root());
    output::detail("Downloads", if args.no_downloads { "disabled" } else { "ask first" });

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
        let options = profiles
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        if let Some(index) = output::prompt_select(
            "What do you want to set up first?",
            &options,
            0,
        )? {
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
        output::detail("JavaScript", "bun install");
        output::detail("Sample model", SAMPLE_MODEL_URL);
        return Ok(());
    }

    if downloads.contains(&SetupDownload::ManagedToolchains) {
        install_managed_toolchains(sh, ctx, profile)?;
    }
    if downloads.contains(&SetupDownload::JavaScriptDependencies) {
        install_javascript_dependencies(sh, ctx)?;
    }
    if downloads.contains(&SetupDownload::SampleModel) {
        download_sample_model(sh, ctx)?;
    }

    Ok(())
}

fn install_managed_toolchains(
    sh: &Shell,
    ctx: &BuildContext,
    profile: SetupProfile,
) -> Result<()> {
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

fn install_javascript_dependencies(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("JavaScript dependencies");
    {
        let _dir = sh.push_dir(ctx.workspace_root());
        output::run_command("Installing workspace Bun dependencies", cmd!(sh, "bun install"))?;
    }
    {
        let _dir = sh.push_dir(ctx.bindings_node_dir());
        output::run_command("Installing Node binding Bun dependencies", cmd!(sh, "bun install"))?;
    }
    Ok(())
}

fn download_sample_model(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    output::phase("Sample model");
    let model_dir = ctx.sample_models_dir();
    sh.create_dir(&model_dir)
        .with_context(|| format!("failed to create {}", model_dir.display()))?;
    let model_path = sample_model_path(ctx);
    if model_path.exists() {
        output::success(format!("Using existing sample model at {}", model_path.display()));
        return Ok(());
    }

    output::detail("Download", SAMPLE_MODEL_URL);
    output::path("Destination", &model_path);
    output::run_command(
        "Downloading sample GGUF model",
        cmd!(sh, "curl -f -L -o {model_path} {SAMPLE_MODEL_URL}"),
    )
}

fn print_examples(ctx: &BuildContext, profile: SetupProfile) {
    output::phase("Next commands");
    output::detail("Doctor", "clm doctor");
    match profile {
        SetupProfile::Browser => {
            output::detail("Build browser package", "clm build wasm");
            output::detail("Run examples app", "clm run apps serve examples");
            output::detail("Run app tests", "clm run apps test");
        }
        SetupProfile::Bindings => {
            let model = model_arg(ctx);
            output::detail("Build Node bindings", "clm build node");
            output::detail("Build Python bindings", "clm build python");
            output::detail("Node smoke", format!("clm run bindings node --model {model}"));
            output::detail(
                "Python smoke",
                format!("clm run bindings python --model {model}"),
            );
        }
        SetupProfile::Full => {
            let model = model_arg(ctx);
            output::detail("Build everything", "clm build all");
            output::detail("Serve benchmark", "clm run apps serve benchmark");
            output::detail("Binding smokes", format!("clm run bindings all --model {model}"));
        }
    }
    output::detail("Compatibility", "cargo xtask still works for every command");
}

fn model_arg(ctx: &BuildContext) -> String {
    let model_path = sample_model_path(ctx);
    if model_path.exists() {
        model_path.display().to_string()
    } else {
        "<model.gguf>".to_owned()
    }
}

fn sample_model_path(ctx: &BuildContext) -> PathBuf {
    ctx.sample_models_dir().join(SAMPLE_MODEL_FILE)
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
            SetupDownload::JavaScriptDependencies => "Install Bun workspace dependencies",
            SetupDownload::SampleModel => "Download the small Qwen sample GGUF model",
        })
    }
}
