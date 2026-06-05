//! Tests the `setup` module in `xtask`.
//!
//! Covers noninteractive profile selection, download recommendations, launcher
//! defaults, and display text using deterministic arguments without prompting
//! or downloading toolchains.

use crate::cli::{SetupArgs, SetupProfile};
use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{
    recommended_downloads, run_downloads, select_downloads, select_profile,
    should_install_launcher, SetupDownload,
};

fn args(
    profile: Option<SetupProfile>,
    yes: bool,
    no_downloads: bool,
    no_splash: bool,
) -> SetupArgs {
    SetupArgs {
        profile,
        yes,
        no_downloads,
        no_splash,
    }
}

#[test]
fn profile_selection_uses_explicit_value_or_browser_default_when_noninteractive() {
    assert_eq!(
        select_profile(&args(Some(SetupProfile::Full), false, false, false), false).unwrap(),
        SetupProfile::Full
    );
    assert_eq!(
        select_profile(&args(None, false, false, false), false).unwrap(),
        SetupProfile::Browser
    );
}

#[test]
fn launcher_install_defaults_to_true_without_prompting() {
    assert!(should_install_launcher(&args(None, true, false, false), false).unwrap());
    assert!(should_install_launcher(&args(None, false, false, false), false).unwrap());
}

#[test]
fn download_selection_respects_no_downloads_yes_and_noninteractive_defaults() {
    assert!(
        select_downloads(SetupProfile::Full, &args(None, false, true, false), false)
            .unwrap()
            .is_empty()
    );
    assert!(
        select_downloads(SetupProfile::Full, &args(None, false, false, false), false)
            .unwrap()
            .is_empty()
    );
    assert_eq!(
        select_downloads(
            SetupProfile::Bindings,
            &args(None, true, false, false),
            false
        )
        .unwrap(),
        vec![
            SetupDownload::ManagedToolchains,
            SetupDownload::JavaScriptDependencies,
            SetupDownload::SampleModel
        ]
    );
}

#[test]
fn recommended_downloads_match_profile_scope() {
    assert_eq!(
        recommended_downloads(SetupProfile::Browser),
        vec![
            SetupDownload::ManagedToolchains,
            SetupDownload::JavaScriptDependencies
        ]
    );
    assert!(recommended_downloads(SetupProfile::Full).contains(&SetupDownload::SampleModel));
}

#[test]
fn setup_download_display_text_is_actionable() {
    assert_eq!(
        SetupDownload::ManagedToolchains.to_string(),
        "Download or activate missing managed toolchains"
    );
    assert_eq!(
        SetupDownload::JavaScriptDependencies.to_string(),
        "Install JavaScript workspace dependencies"
    );
    assert_eq!(
        SetupDownload::SampleModel.to_string(),
        "Download the small Qwen sample GGUF model"
    );
}

#[test]
fn empty_download_run_reports_skip_guidance_without_side_effectful_commands() {
    let temp = TempDir::new("setup-empty-downloads");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let sh = xshell::Shell::new().unwrap();

    run_downloads(&sh, &ctx, SetupProfile::Browser, &[]).unwrap();
}
