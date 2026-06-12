//! WebAssembly/WebGPU browser build target.

use crate::javascript;
use crate::output;
use crate::toolchains::emsdk::{run_with_emsdk, setup_emsdk};
use crate::toolchains::ninja::setup_ninja;
use crate::utils::BuildContext;
use anyhow::Result;
use std::path::Path;
use std::time::Instant;
use xshell::{cmd, Shell};

/// Builds the browser WASM artifacts and TypeScript package wrappers.
pub fn build(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let started_at = Instant::now();
    let root = ctx.workspace_root();
    output::phase("Browser WASM/WebGPU package");
    output::path("Workspace", root);
    output::path("WASM artifact directory", &ctx.npm_browser_wasm_dir());

    let emsdk_dir = setup_emsdk(sh, ctx)?;
    let ninja_dir = setup_ninja(sh, ctx)?;

    let npm_dist_wasm = ctx.npm_browser_wasm_dir();
    sh.create_dir(&npm_dist_wasm)?;

    output::phase("WASM single-thread build");
    build_target(
        sh,
        ctx,
        root,
        &emsdk_dir,
        ninja_dir.as_deref(),
        false,
        &npm_dist_wasm,
    )?;

    output::phase("WASM pthread build");
    build_target(
        sh,
        ctx,
        root,
        &emsdk_dir,
        ninja_dir.as_deref(),
        true,
        &npm_dist_wasm,
    )?;

    output::phase("TypeScript browser package");
    let npm_workspace = ctx.browser_package_dir();
    output::path("Browser package workspace", &npm_workspace);

    javascript::install_root_workspace_dependencies(
        sh,
        ctx,
        "Installing browser package dependencies",
        &[npm_workspace.clone()],
    )?;

    let _npm_dir = sh.push_dir(&npm_workspace);
    output::run_build_command(
        "Compiling TypeScript wrappers",
        cmd!(sh, "bun run build:ts"),
    )?;
    output::run_build_command("Staging browser package", cmd!(sh, "bun run build:stage"))?;

    output::success(format!(
        "WASM pipeline complete in {}",
        output::elapsed(started_at.elapsed())
    ));

    Ok(())
}

fn build_target(
    sh: &Shell,
    ctx: &BuildContext,
    root: &Path,
    emsdk_dir: &Path,
    ninja_dir: Option<&Path>,
    use_pthreads: bool,
    npm_dist_wasm: &Path,
) -> Result<()> {
    let _root_dir = sh.push_dir(root);

    let suffix = if use_pthreads { "-pthread" } else { "" };
    let artifact_name = format!("sipp-wasm{}", suffix);
    let js_file = format!("{}.js", artifact_name);
    let wasm_file = format!("{}.wasm", artifact_name);
    let cargo_target_dir = ctx.cargo_wasm_target_dir(use_pthreads);
    let rust_staticlib = cargo_target_dir
        .join("wasm32-unknown-emscripten")
        .join("release")
        .join("libsipp_wasm.a");

    let rustflags = if use_pthreads {
        "-C target-feature=+atomics,+bulk-memory,+mutable-globals"
    } else {
        ""
    };
    let cargo_cmd = if cfg!(windows) {
        format!(
            "set RUSTFLAGS={rustflags}\r\ncargo build --release --package sipp-wasm --target wasm32-unknown-emscripten --target-dir {}",
            ctx.command_path(&cargo_target_dir)
        )
    } else if rustflags.is_empty() {
        format!(
            "cargo build --release --package sipp-wasm --target wasm32-unknown-emscripten --target-dir {}",
            ctx.command_path(&cargo_target_dir)
        )
    } else {
        format!(
            "RUSTFLAGS='{rustflags}' cargo build --release --package sipp-wasm --target wasm32-unknown-emscripten --target-dir {}",
            ctx.command_path(&cargo_target_dir)
        )
    };

    run_with_emsdk(
        sh,
        emsdk_dir,
        ninja_dir,
        &format!("Compiling Rust staticlib for {js_file}"),
        &cargo_cmd,
    )?;

    output::step("Linking browser runtime via Emscripten");
    let wasm_dir = root.join("bindings").join("wasm");
    let wasm_source_dir = ctx.cmake_file_path(&wasm_dir);
    let rust_staticlib_cmake = ctx.cmake_file_path(&rust_staticlib);
    let build_dir = ctx.cmake_wasm_build_dir(use_pthreads);
    sh.create_dir(&build_dir)?;
    output::path("CMake build directory", &build_dir);

    let _dir = sh.push_dir(&build_dir);

    let cmake_thread_flag = if use_pthreads {
        "-DCE_USE_PTHREADS=ON"
    } else {
        "-DCE_USE_PTHREADS=OFF"
    };
    let emcmake_cmd = format!(
        "emcmake cmake \"{}\" -G Ninja -DCMAKE_BUILD_TYPE=Release {} -DCE_WASM_RUST_STATICLIB=\"{}\"",
        wasm_source_dir, cmake_thread_flag, rust_staticlib_cmake
    );
    run_with_emsdk(
        sh,
        emsdk_dir,
        ninja_dir,
        &format!("Configuring CMake for {artifact_name}"),
        &emcmake_cmd,
    )?;

    let build_cmd = "cmake --build . --parallel";
    run_with_emsdk(
        sh,
        emsdk_dir,
        ninja_dir,
        &format!("Building browser runtime for {artifact_name}"),
        build_cmd,
    )?;
    drop(_dir);

    let compiled_js = build_dir.join("dist").join("Sipp.js");
    let compiled_wasm = build_dir.join("dist").join("Sipp.wasm");

    let staged_js = npm_dist_wasm.join(&js_file);
    let staged_wasm = npm_dist_wasm.join(&wasm_file);
    sh.copy_file(&compiled_js, &staged_js)?;
    sh.copy_file(&compiled_wasm, &staged_wasm)?;
    output::artifact(&staged_js);
    output::artifact(&staged_wasm);

    Ok(())
}
