//! Build and serve the documentation book with mermaid diagram support.
//!
//! Ensures `mdbook` and `mdbook-mermaid` are installed, generates a minimal
//! theme CSS and extracts the bundled mermaid JavaScript assets into `theme/`
//! at the workspace root (gitignored), then builds or serves the book.

use crate::output;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::fs;
use xshell::{cmd, Shell};

const MDBOOK_VERSION: &str = "0.5.3";

const COGENTLM_CSS: &str = r#":root {
    --content-max-width: 900px;
}

header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: #e0e0e0;
}

header .title {
    font-weight: 700;
    letter-spacing: 0.02em;
}
"#;

const MINIMAL_BOOK_TOML: &str = r#"[book]
src = "."

[output.html]
"#;

/// Build the documentation book.
pub fn run_build(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    ensure_deps(sh)?;
    ensure_theme_assets(sh, ctx)?;

    output::phase("mdBook build");
    let _dir = sh.push_dir(ctx.workspace_root());
    cmd!(sh, "mdbook build").run()?;

    output::success("Documentation built");
    Ok(())
}

/// Build and serve the documentation book with live reload.
pub fn run_serve(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    ensure_deps(sh)?;
    ensure_theme_assets(sh, ctx)?;

    output::phase("mdBook serve");
    let _dir = sh.push_dir(ctx.workspace_root());
    cmd!(sh, "mdbook serve --open").run()?;

    Ok(())
}

/// Install `mdbook` and `mdbook-mermaid` via cargo when not already installed.
fn ensure_deps(sh: &Shell) -> Result<()> {
    output::phase("Checking docs dependencies");

    let mdbook_installed = sh
        .cmd("mdbook")
        .arg("--version")
        .ignore_stderr()
        .ignore_stdout()
        .run()
        .is_ok();

    if !mdbook_installed {
        output::step("Installing mdbook");
        output::run_command(
            "cargo install mdbook",
            cmd!(
                sh,
                "cargo install mdbook --version {MDBOOK_VERSION} --locked"
            ),
        )?;
        output::success("mdbook installed");
    } else {
        output::detail("mdbook", "already installed");
    }

    let mermaid_installed = sh
        .cmd("mdbook-mermaid")
        .arg("--version")
        .ignore_stderr()
        .ignore_stdout()
        .run()
        .is_ok();

    if !mermaid_installed {
        output::step("Installing mdbook-mermaid");
        output::run_command(
            "cargo install mdbook-mermaid",
            cmd!(sh, "cargo install mdbook-mermaid --locked"),
        )?;
        output::success("mdbook-mermaid installed");
    } else {
        output::detail("mdbook-mermaid", "already installed");
    }

    Ok(())
}

/// Generate theme CSS and extract mermaid JS assets into `theme/` (workspace
/// root). The CSS is hardcoded; mermaid JS is extracted from the
/// `mdbook-mermaid` crate using a temporary book directory.
fn ensure_theme_assets(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let theme_dir = ctx.workspace_root().join("theme");

    output::phase("Ensuring theme assets");

    fs::create_dir_all(&theme_dir)
        .with_context(|| format!("failed to create {}", theme_dir.display()))?;

    let css_path = theme_dir.join("cogentlm.css");
    if !css_path.exists() {
        output::step("Generating theme/cogentlm.css");
        fs::write(&css_path, COGENTLM_CSS)
            .with_context(|| format!("failed to write {}", css_path.display()))?;
        output::path("Written", &css_path);
    } else {
        output::detail("cogentlm.css", "already present in theme/");
    }

    let mermaid_js = theme_dir.join("mermaid.min.js");
    let mermaid_init = theme_dir.join("mermaid-init.js");

    if mermaid_js.exists() && mermaid_init.exists() {
        output::detail("Mermaid assets", "already present in theme/");
        return Ok(());
    }

    let temp_dir = ctx.build_root().join("tmp").join("docs-mermaid");
    fs::create_dir_all(&temp_dir)
        .with_context(|| format!("failed to create {}", temp_dir.display()))?;

    let temp_book_toml = temp_dir.join("book.toml");
    fs::write(&temp_book_toml, MINIMAL_BOOK_TOML)
        .with_context(|| format!("failed to write {}", temp_book_toml.display()))?;

    output::step("Running mdbook-mermaid install in temporary directory");
    let _dir = sh.push_dir(ctx.workspace_root());
    cmd!(sh, "mdbook-mermaid install {temp_dir}")
        .ignore_stderr()
        .ignore_stdout()
        .run()
        .context("mdbook-mermaid install failed; try `cargo install mdbook-mermaid`")?;

    output::step("Copying mermaid assets to theme/");
    let temp_js = temp_dir.join("mermaid.min.js");
    let temp_init = temp_dir.join("mermaid-init.js");

    if temp_js.exists() {
        fs::copy(&temp_js, &mermaid_js)
            .with_context(|| format!("failed to copy {} to theme/", temp_js.display()))?;
        output::path("Copied", &mermaid_js);
    } else {
        anyhow::bail!(
            "mdbook-mermaid did not produce mermaid.min.js in {}",
            temp_dir.display()
        );
    }

    if temp_init.exists() {
        fs::copy(&temp_init, &mermaid_init)
            .with_context(|| format!("failed to copy {} to theme/", temp_init.display()))?;
        output::path("Copied", &mermaid_init);
    } else {
        anyhow::bail!(
            "mdbook-mermaid did not produce mermaid-init.js in {}",
            temp_dir.display()
        );
    }

    fs::remove_dir_all(&temp_dir)
        .with_context(|| format!("failed to clean up {}", temp_dir.display()))?;

    Ok(())
}
