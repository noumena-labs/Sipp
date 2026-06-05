//! Shared build paths and formatting helpers.

use crate::cli::Backend;
use crate::output;
use anyhow::{Context, Result};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;
use xshell::{cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/utils_tests.rs"]
mod utils_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const BROWSER_PACKAGE_ARTIFACT_DIR: &str = "cogentlm";

/// Shared immutable context for xtask build paths.
#[derive(Clone, Debug)]
pub struct BuildContext {
    workspace_root: PathBuf,
}

impl BuildContext {
    /// Creates a build context rooted at the Cargo workspace.
    pub fn new() -> Result<Self> {
        let manifest_dir = PathBuf::from(
            env::var("CARGO_MANIFEST_DIR")
                .unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned()),
        );
        let workspace_root = manifest_dir
            .parent()
            .map(Path::to_path_buf)
            .with_context(|| {
                format!(
                    "failed to resolve workspace root from {}",
                    manifest_dir.display()
                )
            })?;

        Ok(Self { workspace_root })
    }

    #[cfg(test)]
    pub(crate) fn from_workspace_root_for_test(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
        }
    }

    pub(crate) fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub(crate) fn build_root(&self) -> PathBuf {
        self.workspace_root.join(".build")
    }

    pub(crate) fn cargo_build_root(&self) -> PathBuf {
        self.build_root().join("cargo")
    }

    pub(crate) fn cmake_build_root(&self) -> PathBuf {
        self.build_root().join("cmake")
    }

    pub(crate) fn native_build_root(&self) -> PathBuf {
        self.build_root().join("c")
    }

    pub(crate) fn cargo_node_target_dir(&self, backend: &Backend) -> PathBuf {
        self.cargo_build_root().join("node").join(backend.as_str())
    }

    pub(crate) fn cargo_cli_target_dir(&self, backend: &Backend) -> PathBuf {
        self.cargo_build_root().join("cli").join(backend.as_str())
    }

    pub(crate) fn cargo_python_target_dir(&self, backend: Option<&Backend>) -> PathBuf {
        self.cargo_build_root()
            .join("python")
            .join(Self::backend_build_tag(backend))
    }

    pub(crate) fn cargo_wasm_target_dir(&self, use_pthreads: bool) -> PathBuf {
        self.cargo_build_root()
            .join("wasm")
            .join(Self::wasm_build_tag(use_pthreads))
    }

    pub(crate) fn cmake_wasm_build_dir(&self, use_pthreads: bool) -> PathBuf {
        self.build_root()
            .join("cmake")
            .join("wasm")
            .join(Self::wasm_build_tag(use_pthreads))
    }

    pub(crate) fn cmake_llama_build_dir(&self, backend: &Backend) -> PathBuf {
        self.cmake_build_root().join("llama").join(backend.as_str())
    }

    pub(crate) fn cmake_cli_sys_dir(&self, backend: &Backend) -> PathBuf {
        if cfg!(windows) {
            self.native_build_root().join("cli").join(backend.as_str())
        } else {
            self.cmake_build_root()
                .join("cli-sys")
                .join(backend.as_str())
        }
    }

    pub(crate) fn artifacts_root(&self) -> PathBuf {
        self.build_root().join("artifacts")
    }

    pub(crate) fn node_artifacts_dir(&self) -> PathBuf {
        self.artifacts_root().join("node")
    }

    pub(crate) fn python_artifacts_dir(&self) -> PathBuf {
        self.artifacts_root().join("python")
    }

    pub(crate) fn cli_artifacts_dir(&self) -> PathBuf {
        self.artifacts_root().join("cli")
    }

    pub(crate) fn npm_browser_artifacts_dir(&self) -> PathBuf {
        self.artifacts_root()
            .join("npm")
            .join(BROWSER_PACKAGE_ARTIFACT_DIR)
    }

    pub(crate) fn npm_browser_wasm_dir(&self) -> PathBuf {
        self.npm_browser_artifacts_dir().join("dist").join("wasm")
    }

    pub(crate) fn demo_artifacts_dir(&self, demo: &str) -> PathBuf {
        self.artifacts_root().join("demos").join(demo)
    }

    pub(crate) fn example_artifacts_dir(&self, example: &str) -> PathBuf {
        self.artifacts_root().join("examples").join(example)
    }

    pub(crate) fn benchmark_artifacts_dir(&self, benchmark: &str) -> PathBuf {
        self.artifacts_root().join("benchmarks").join(benchmark)
    }

    pub(crate) fn toolchain_dir(&self) -> PathBuf {
        self.build_root().join("toolchain")
    }

    pub(crate) fn tmp_dir(&self) -> PathBuf {
        self.build_root().join("tmp")
    }

    pub(crate) fn command_logs_dir(&self) -> PathBuf {
        self.build_root().join("logs")
    }

    pub(crate) fn launcher_bin_dir(&self) -> PathBuf {
        self.build_root().join("bin")
    }

    pub(crate) fn sample_models_dir(&self) -> PathBuf {
        self.build_root().join("models")
    }

    pub(crate) fn lib_root(&self) -> PathBuf {
        self.workspace_root.join("lib")
    }

    pub(crate) fn demos_root(&self) -> PathBuf {
        self.workspace_root.join("demos")
    }

    pub(crate) fn examples_root(&self) -> PathBuf {
        self.workspace_root.join("examples")
    }

    pub(crate) fn benchmarks_root(&self) -> PathBuf {
        self.workspace_root.join("benchmarks")
    }

    pub(crate) fn bindings_node_dir(&self) -> PathBuf {
        self.workspace_root.join("bindings").join("node")
    }

    pub(crate) fn bindings_python_dir(&self) -> PathBuf {
        self.workspace_root.join("bindings").join("python")
    }

    pub(crate) fn browser_package_dir(&self) -> PathBuf {
        self.lib_root().join("web")
    }

    pub(crate) fn node_package_dir(&self) -> PathBuf {
        self.lib_root().join("node")
    }

    pub(crate) fn python_package_project_dir(&self) -> PathBuf {
        self.lib_root().join("python")
    }

    fn playwright_core_cli(&self) -> PathBuf {
        self.workspace_root
            .join("node_modules")
            .join("playwright-core")
            .join("cli.js")
    }

    pub(crate) fn playwright_browsers_dir(&self) -> PathBuf {
        self.toolchain_dir().join("playwright-browsers")
    }

    fn playwright_chromium_executable(&self) -> Result<(PathBuf, bool)> {
        let browsers_dir = self.playwright_browsers_dir();
        let output = Command::new("node")
            .arg("-e")
            .arg(
                "const fs = require('node:fs'); \
                 const { chromium } = require('playwright-core'); \
                 const executable = chromium.executablePath(); \
                 console.log(executable); \
                 console.log(fs.existsSync(executable) ? 'true' : 'false');",
            )
            .current_dir(&self.workspace_root)
            .env("PLAYWRIGHT_BROWSERS_PATH", &browsers_dir)
            .output()
            .context("failed to query Playwright Chromium executable path")?;
        if !output.status.success() {
            anyhow::bail!(
                "failed to query Playwright Chromium executable path: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }

        let stdout = String::from_utf8(output.stdout)?;
        let mut lines = stdout.lines();
        let executable = lines
            .next()
            .context("Playwright did not print a Chromium executable path")?;
        let installed = lines.next() == Some("true");
        Ok((PathBuf::from(executable), installed))
    }

    pub(crate) fn llama_cpp_dir(&self) -> PathBuf {
        self.workspace_root.join("third_party").join("llama.cpp")
    }

    pub(crate) fn demo_dir(&self, demo: &str) -> PathBuf {
        self.demos_root().join(demo)
    }

    pub(crate) fn demo_dirs(&self) -> Result<Vec<PathBuf>> {
        read_child_dirs(&self.demos_root())
    }

    pub(crate) fn browser_example_dir(&self) -> PathBuf {
        self.examples_root().join("web")
    }

    pub(crate) fn benchmark_browser_dir(&self) -> PathBuf {
        self.benchmarks_root().join("browser")
    }

    pub(crate) fn benchmark_dirs(&self) -> Result<Vec<PathBuf>> {
        read_child_dirs(&self.benchmarks_root())
    }

    pub(crate) fn js_package_dirs(&self) -> Vec<PathBuf> {
        vec![self.browser_package_dir(), self.node_package_dir()]
    }

    pub(crate) fn uv_toolchain_dir(&self) -> PathBuf {
        self.toolchain_dir().join("uv")
    }

    pub(crate) fn uv_exe(&self) -> PathBuf {
        self.uv_toolchain_dir()
            .join(if cfg!(windows) { "uv.exe" } else { "uv" })
    }

    pub(crate) fn ninja_toolchain_dir(&self) -> PathBuf {
        self.toolchain_dir().join("ninja")
    }

    pub(crate) fn ninja_exe(&self) -> PathBuf {
        self.ninja_toolchain_dir()
            .join(if cfg!(windows) { "ninja.exe" } else { "ninja" })
    }

    pub(crate) fn emsdk_dir(&self) -> PathBuf {
        self.toolchain_dir().join("emsdk")
    }

    pub(crate) fn vulkan_dir(&self) -> PathBuf {
        self.toolchain_dir().join("vulkan")
    }

    pub(crate) fn backend_build_tag(backend: Option<&Backend>) -> &'static str {
        backend.map(Backend::as_str).unwrap_or("cpu")
    }

    pub(crate) fn wasm_build_tag(use_pthreads: bool) -> &'static str {
        if use_pthreads {
            "pthread"
        } else {
            "single"
        }
    }

    pub(crate) fn command_path(&self, path: &Path) -> String {
        format!("\"{}\"", path.display())
    }

    pub(crate) fn cmake_file_path(&self, path: &Path) -> String {
        path.display().to_string().replace('\\', "/")
    }
}

pub(crate) fn ensure_playwright_chromium(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let (chromium_exe, installed) = ctx.playwright_chromium_executable()?;
    if installed {
        output::detail("Playwright Chromium", chromium_exe.display());
        return Ok(());
    }

    let playwright_cli = ctx.playwright_core_cli();
    if !playwright_cli.is_file() {
        anyhow::bail!(
            "Playwright Core CLI was not found at {}; run `bun install` at the workspace root",
            playwright_cli.display()
        );
    }

    let _dir = sh.push_dir(ctx.workspace_root());
    let browsers_dir = ctx.playwright_browsers_dir();
    output::run_command(
        "Installing Playwright Chromium",
        cmd!(sh, "node {playwright_cli} install chromium")
            .env("PLAYWRIGHT_BROWSERS_PATH", browsers_dir),
    )
}

fn read_child_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut dirs = Vec::new();
    for entry in
        std::fs::read_dir(root).with_context(|| format!("failed to read {}", root.display()))?
    {
        let path = entry?.path();
        if path.is_dir() {
            dirs.push(path);
        }
    }
    dirs.sort();
    Ok(dirs)
}
