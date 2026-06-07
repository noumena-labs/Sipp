//! Workspace cleanup commands.

use crate::cli::CleanArgs;
use crate::output;
use crate::utils::BuildContext;
use anyhow::{Context, Result};
use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use xshell::Shell;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "tests/clean_tests.rs"]
mod clean_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

const GENERATED_DIRS: &[&str] = &["dist", ".vite", "coverage", "build", "out", ".next"];

/// Runs the workspace cleanup command.
pub fn run(sh: &Shell, ctx: &BuildContext, args: &CleanArgs) -> Result<()> {
    output::phase("Clean workspace");
    output::path("Workspace", ctx.workspace_root());

    let targets = clean_targets(ctx, args)?;
    if args.dry_run {
        output::detail("Mode", "dry run");
    }
    if args.purge {
        output::warning("Purge enabled: workspace node_modules directories are included");
    }
    if args.toolchains {
        output::warning("Toolchain cleanup enabled: .build/toolchain is included");
    }

    if targets.is_empty() {
        output::success("Nothing to clean");
        return Ok(());
    }

    let workspace = canonical_workspace(ctx)?;
    for target in targets {
        if !target.exists() {
            output::detail("Already clean", target.display());
            continue;
        }

        let checked = checked_delete_target(&workspace, &target)?;
        if args.dry_run {
            output::path("Would remove", &checked);
            continue;
        }

        output::path("Removing", &checked);
        sh.remove_path(&checked)
            .with_context(|| format!("failed to remove {}", checked.display()))?;
    }

    if args.dry_run {
        output::success("Dry run complete");
    } else {
        output::success("Workspace clean complete");
    }

    Ok(())
}

fn clean_targets(ctx: &BuildContext, args: &CleanArgs) -> Result<Vec<PathBuf>> {
    let mut targets = BTreeSet::new();

    add_cargo_clean_targets(&mut targets, ctx)?;
    targets.insert(ctx.cmake_build_root());
    targets.insert(ctx.native_build_root());
    targets.insert(ctx.artifacts_root());
    targets.insert(ctx.tmp_dir());
    targets.insert(ctx.command_logs_dir());
    targets.insert(ctx.browser_package_dir().join("dist"));

    for dir in ctx.demo_dirs()? {
        add_generated_dirs(&mut targets, &dir);
    }
    for dir in ctx.tool_dirs()? {
        add_generated_dirs(&mut targets, &dir);
    }
    for dir in ctx.js_package_dirs() {
        add_generated_dirs(&mut targets, &dir);
    }

    if args.purge {
        targets.insert(ctx.workspace_root().join("node_modules"));
        targets.insert(ctx.bindings_node_dir().join("node_modules"));

        for dir in ctx.demo_dirs()? {
            targets.insert(dir.join("node_modules"));
        }
        for dir in ctx.tool_dirs()? {
            targets.insert(dir.join("node_modules"));
        }
        for dir in ctx.js_package_dirs() {
            targets.insert(dir.join("node_modules"));
        }
    }

    if args.toolchains {
        targets.insert(ctx.toolchain_dir());
    }

    Ok(targets.into_iter().collect())
}

fn add_cargo_clean_targets(targets: &mut BTreeSet<PathBuf>, ctx: &BuildContext) -> Result<()> {
    let cargo_root = ctx.cargo_build_root();

    let Some(protected_target) = protected_cargo_target(&cargo_root)? else {
        targets.insert(cargo_root);
        return Ok(());
    };

    output::warning(format!(
        "Preserving current xtask target: {}",
        protected_target.display()
    ));

    if !cargo_root.exists() {
        return Ok(());
    }

    for entry in fs::read_dir(&cargo_root)
        .with_context(|| format!("failed to read {}", cargo_root.display()))?
    {
        let path = entry?.path();
        if paths_match(&path, &protected_target)? {
            continue;
        }

        targets.insert(path);
    }

    Ok(())
}

fn protected_cargo_target(cargo_root: &Path) -> Result<Option<PathBuf>> {
    if !cfg!(windows) || !cargo_root.exists() {
        return Ok(None);
    }

    let cargo_root = cargo_root
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", cargo_root.display()))?;
    let current_exe = env::current_exe()
        .context("failed to resolve current xtask executable")?
        .canonicalize()
        .context("failed to canonicalize current xtask executable")?;

    Ok(first_child_under(&cargo_root, &current_exe))
}

fn first_child_under(root: &Path, path: &Path) -> Option<PathBuf> {
    let mut components = path.strip_prefix(root).ok()?.components();
    let first = components.next()?;

    Some(root.join(first.as_os_str()))
}

fn paths_match(left: &Path, right: &Path) -> Result<bool> {
    let left = left
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", left.display()))?;
    let right = right
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", right.display()))?;

    Ok(left == right)
}

fn add_generated_dirs(targets: &mut BTreeSet<PathBuf>, root: &Path) {
    for name in GENERATED_DIRS {
        targets.insert(root.join(name));
    }
}

fn canonical_workspace(ctx: &BuildContext) -> Result<PathBuf> {
    ctx.workspace_root()
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", ctx.workspace_root().display()))
}

fn checked_delete_target(workspace: &Path, target: &Path) -> Result<PathBuf> {
    let canonical = target
        .canonicalize()
        .with_context(|| format!("failed to resolve {}", target.display()))?;

    if !canonical.starts_with(workspace) {
        anyhow::bail!(
            "refusing to delete {} because it is outside workspace {}",
            canonical.display(),
            workspace.display()
        );
    }

    if canonical == workspace {
        anyhow::bail!("refusing to delete workspace root {}", workspace.display());
    }

    Ok(canonical)
}
