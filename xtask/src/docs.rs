//! Build and serve the documentation book with mermaid diagram support.
//!
//! Ensures `mdbook` and `mdbook-mermaid` are installed, writes the xtask-owned
//! theme CSS and mermaid loader into `theme/` at the workspace root
//! (gitignored), extracts the bundled `mermaid.min.js` from `mdbook-mermaid`,
//! then builds or serves the book. `mermaid.min.js` is staged into each built
//! tree and fetched on demand by the loader instead of loading on every page.

use crate::cli::DocsLanguage;
use crate::output;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::fs;
use xshell::{cmd, Cmd, Shell};

const MDBOOK_VERSION: &str = "0.5.3";
const DOCS_BUILD_DIR: &str = "book";
const ZH_DOCS_BUILD_DIR: &str = "book/zh";
const DOCS_SERVE_URL: &str = "http://localhost:3000";

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

/* Crossfade same-origin navigations (e.g. the language switch) instead of a
   white flash. Browsers without view-transition support ignore this. */
@view-transition {
    navigation: auto;
}
"#;

const MERMAID_INIT_JS: &str = r#"(() => {
    // Mermaid is ~2.7 MB; only fetch it on pages that contain a diagram.
    if (document.querySelector('.mermaid') == null) {
        return;
    }

    const darkThemes = ['ayu', 'navy', 'coal'];
    const isDarkTheme = () =>
        darkThemes.some((theme) => document.documentElement.classList.contains(theme));

    const renderedDark = isDarkTheme();
    const script = document.createElement('script');
    script.src = `${path_to_root}theme/mermaid.min.js`;
    script.onload = () => {
        mermaid.initialize({ startOnLoad: false, theme: renderedDark ? 'dark' : 'default' });
        mermaid.run();
    };
    document.head.appendChild(script);

    // Mermaid bakes colors into the rendered SVG, so re-render via reload when
    // the page switches between light and dark themes.
    for (const button of document.querySelectorAll('#mdbook-theme-list .theme')) {
        button.addEventListener('click', () => {
            setTimeout(() => {
                if (isDarkTheme() !== renderedDark) {
                    window.location.reload();
                }
            });
        });
    }
})();
"#;

const DOCS_SERVE_PY: &str = r#"import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HASHED_ASSET = re.compile(r"-[0-9a-f]{8,}\.[A-Za-z0-9.]+$")


class CachingHandler(SimpleHTTPRequestHandler):
    """Serve mdBook output with cache headers suited to hashed assets."""

    def end_headers(self):
        path = self.path.split("?", 1)[0]
        if HASHED_ASSET.search(path):
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()


def main():
    handler = partial(CachingHandler, directory=sys.argv[1])
    ThreadingHTTPServer(("127.0.0.1", 3000), handler).serve_forever()


main()
"#;

const MINIMAL_BOOK_TOML: &str = r#"[book]
src = "."

[output.html]
"#;

/// Build the documentation book.
pub fn run_build(sh: &Shell, ctx: &BuildContext, lang: DocsLanguage) -> Result<()> {
    ensure_deps(sh)?;
    ensure_theme_assets(sh, ctx)?;

    let _dir = sh.push_dir(ctx.workspace_root());

    output::phase(&format!("mdBook build ({})", lang.as_str()));
    mdbook_build_cmd(sh, lang).run()?;

    if lang != DocsLanguage::Zh {
        output::phase("mdBook build (zh)");
        mdbook_build_cmd(sh, DocsLanguage::Zh).run()?;
    }

    stage_mermaid_runtime(ctx)?;

    output::success("Documentation built");
    Ok(())
}

/// Build and serve the documentation book.
pub fn run_serve(sh: &Shell, ctx: &BuildContext, lang: DocsLanguage) -> Result<()> {
    ensure_deps(sh)?;
    ensure_theme_assets(sh, ctx)?;

    let _dir = sh.push_dir(ctx.workspace_root());

    output::phase("mdBook build (en)");
    mdbook_build_cmd(sh, DocsLanguage::En).run()?;

    output::phase("mdBook build (zh)");
    mdbook_build_cmd(sh, DocsLanguage::Zh).run()?;

    stage_mermaid_runtime(ctx)?;

    let url = if lang == DocsLanguage::Zh {
        format!("{DOCS_SERVE_URL}/zh/")
    } else {
        DOCS_SERVE_URL.to_string()
    };

    output::phase("Serving documentation");
    output::detail("URL", url);
    serve_book_cmd(sh).run()?;

    Ok(())
}

fn mdbook_build_cmd<'a>(sh: &'a Shell, lang: DocsLanguage) -> Cmd<'a> {
    apply_language_env(cmd!(sh, "mdbook build"), lang)
}

fn apply_language_env<'a>(cmd: Cmd<'a>, lang: DocsLanguage) -> Cmd<'a> {
    if lang == DocsLanguage::Zh {
        return cmd
            .env("MDBOOK_BOOK__SRC", "docs_zh")
            .env("MDBOOK_BOOK__LANGUAGE", DocsLanguage::Zh.as_str())
            .env("MDBOOK_BOOK__TITLE", "CogentLM 中文文档")
            .env("MDBOOK_BUILD__BUILD_DIR", ZH_DOCS_BUILD_DIR);
    }

    cmd
}

fn serve_book_cmd<'a>(sh: &'a Shell) -> Cmd<'a> {
    let python = if cfg!(windows) { "python" } else { "python3" };
    cmd!(sh, "{python} -c {DOCS_SERVE_PY} {DOCS_BUILD_DIR}")
}

/// Copy `theme/mermaid.min.js` into each built tree. The file is fetched on
/// demand by `mermaid-init.js` rather than listed in `additional-js`, so
/// mdBook does not stage it.
fn stage_mermaid_runtime(ctx: &BuildContext) -> Result<()> {
    let source = ctx.workspace_root().join("theme").join("mermaid.min.js");
    for tree in [DOCS_BUILD_DIR, ZH_DOCS_BUILD_DIR] {
        let tree_dir = ctx.workspace_root().join(tree);
        if !tree_dir.exists() {
            continue;
        }
        let target_dir = tree_dir.join("theme");
        fs::create_dir_all(&target_dir)
            .with_context(|| format!("failed to create {}", target_dir.display()))?;
        let target = target_dir.join("mermaid.min.js");
        fs::copy(&source, &target)
            .with_context(|| format!("failed to copy mermaid.min.js to {}", target.display()))?;
    }
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

/// Write the xtask-owned theme assets and extract `mermaid.min.js` into
/// `theme/` (workspace root). `cogentlm.css` and `mermaid-init.js` are
/// rewritten every run so edits to the constants in this file reach existing
/// checkouts; `mermaid.min.js` is extracted from the `mdbook-mermaid` crate
/// using a temporary book directory when missing.
fn ensure_theme_assets(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let theme_dir = ctx.workspace_root().join("theme");

    output::phase("Ensuring theme assets");

    fs::create_dir_all(&theme_dir)
        .with_context(|| format!("failed to create {}", theme_dir.display()))?;

    let css_path = theme_dir.join("cogentlm.css");
    fs::write(&css_path, COGENTLM_CSS)
        .with_context(|| format!("failed to write {}", css_path.display()))?;
    output::path("Written", &css_path);

    let mermaid_init = theme_dir.join("mermaid-init.js");
    fs::write(&mermaid_init, MERMAID_INIT_JS)
        .with_context(|| format!("failed to write {}", mermaid_init.display()))?;
    output::path("Written", &mermaid_init);

    let mermaid_js = theme_dir.join("mermaid.min.js");
    if mermaid_js.exists() {
        output::detail("mermaid.min.js", "already present in theme/");
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

    let temp_js = temp_dir.join("mermaid.min.js");
    if !temp_js.exists() {
        anyhow::bail!(
            "mdbook-mermaid did not produce mermaid.min.js in {}",
            temp_dir.display()
        );
    }
    fs::copy(&temp_js, &mermaid_js)
        .with_context(|| format!("failed to copy {} to theme/", temp_js.display()))?;
    output::path("Copied", &mermaid_js);

    fs::remove_dir_all(&temp_dir)
        .with_context(|| format!("failed to clean up {}", temp_dir.display()))?;

    Ok(())
}
