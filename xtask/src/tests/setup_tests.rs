//! Tests the `setup` module in `xtask`.
//!
//! Covers noninteractive profile selection, download recommendations, launcher
//! defaults, and display text using deterministic arguments without prompting
//! or downloading toolchains.

use crate::cli::{SetupArgs, SetupProfile};
use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{
    recommended_downloads, root_javascript_package_dirs, run_downloads, select_downloads,
    select_profile, should_install_launcher, should_install_node_binding_javascript_dependencies,
    SetupDownload,
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
fn javascript_package_dirs_match_profile_scope() {
    let temp = TempDir::new("setup-javascript-scope");
    temp.create_dir("lib/web");
    temp.create_dir("lib/node");
    temp.create_dir("examples/web");
    temp.create_dir("demos/chat");
    temp.create_dir("demos/avatar");
    temp.create_dir("tools/playground");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());

    let browser = root_javascript_package_dirs(&ctx, SetupProfile::Browser).unwrap();
    assert!(browser.contains(&temp.join("lib/web")));
    assert!(browser.contains(&temp.join("examples/web")));
    assert!(browser.contains(&temp.join("demos/avatar")));
    assert!(browser.contains(&temp.join("demos/chat")));
    assert!(browser.contains(&temp.join("tools/playground")));
    assert!(!browser.contains(&temp.join("lib/node")));

    let bindings = root_javascript_package_dirs(&ctx, SetupProfile::Bindings).unwrap();
    assert_eq!(bindings, vec![temp.join("lib/node")]);

    let full = root_javascript_package_dirs(&ctx, SetupProfile::Full).unwrap();
    assert!(full.contains(&temp.join("lib/web")));
    assert!(full.contains(&temp.join("lib/node")));
}

#[test]
fn node_binding_javascript_dependencies_match_setup_profile_scope() {
    assert!(!should_install_node_binding_javascript_dependencies(
        SetupProfile::Browser
    ));
    assert!(should_install_node_binding_javascript_dependencies(
        SetupProfile::Bindings
    ));
    assert!(should_install_node_binding_javascript_dependencies(
        SetupProfile::Full
    ));
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
