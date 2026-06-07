//! JavaScript package-manager helpers for xtask workflows.

use crate::output;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/javascript_tests.rs"]
mod javascript_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

pub(crate) fn install_root_workspace_dependencies(
    sh: &Shell,
    ctx: &BuildContext,
    label: impl Into<String>,
    package_dirs: &[PathBuf],
) -> Result<()> {
    let filters = root_workspace_package_filters(package_dirs)?;
    let _dir = sh.push_dir(ctx.workspace_root());
    let mut install_cmd = cmd!(sh, "bun install --frozen-lockfile");
    for filter in &filters {
        install_cmd = install_cmd.arg("--filter").arg(filter);
    }
    output::run_build_command(label, install_cmd)
}

pub(crate) fn install_node_binding_dependencies(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let _dir = sh.push_dir(ctx.bindings_node_dir());
    output::run_build_command(
        "Installing Node binding JavaScript dependencies",
        cmd!(sh, "bun install --frozen-lockfile"),
    )
}

fn root_workspace_package_filters(package_dirs: &[PathBuf]) -> Result<Vec<String>> {
    anyhow::ensure!(
        !package_dirs.is_empty(),
        "at least one JavaScript workspace package directory is required"
    );

    let mut filters = Vec::new();
    for package_dir in package_dirs {
        let name = package_name(package_dir)?;
        if !filters.contains(&name) {
            filters.push(name);
        }
    }
    Ok(filters)
}

fn package_name(package_dir: &Path) -> Result<String> {
    let package_json = package_dir.join("package.json");
    let manifest = std::fs::read_to_string(&package_json)
        .with_context(|| format!("failed to read {}", package_json.display()))?;
    let value = serde_json::from_str::<Value>(&manifest)
        .with_context(|| format!("failed to parse {}", package_json.display()))?;
    let name = value
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .with_context(|| format!("missing package name in {}", package_json.display()))?;
    Ok(name.to_owned())
}
