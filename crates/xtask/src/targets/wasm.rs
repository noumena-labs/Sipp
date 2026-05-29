//! WebAssembly/WebGPU browser build target.

use crate::toolchains::emsdk::{run_with_emsdk, setup_emsdk};
use crate::toolchains::ninja::setup_ninja;
use crate::utils::BuildContext;
use anyhow::Result;
use std::path::Path;
use xshell::{cmd, Shell};

/// Builds the browser WASM artifacts and TypeScript package wrappers.
pub fn build(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let root = ctx.workspace_root();
    let emsdk_dir = setup_emsdk(sh, ctx)?;
    let ninja_dir = setup_ninja(sh, ctx)?;

    let npm_dist_wasm = ctx.npm_browser_wasm_dir();
    sh.create_dir(&npm_dist_wasm)?;

    println!("=> Starting Phase 1: Single-Threaded Build");
    build_target(
        sh,
        ctx,
        root,
        &emsdk_dir,
        ninja_dir.as_deref(),
        false,
        &npm_dist_wasm,
    )?;

    println!("=> Starting Phase 2: PThread Build");
    build_target(
        sh,
        ctx,
        root,
        &emsdk_dir,
        ninja_dir.as_deref(),
        true,
        &npm_dist_wasm,
    )?;

    println!("=> Starting Phase 3: Compiling TypeScript Wrappers...");
    let npm_workspace = root.join("packages").join("npm");
    let _npm_dir = sh.push_dir(&npm_workspace);

    cmd!(sh, "bun install").run()?;
    cmd!(sh, "bun run build:ts").run()?;
    cmd!(sh, "bun run build:stage").run()?;

    println!("=> WASM Pipeline Complete!");
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
    let artifact_name = format!("cogentlm-wasm{}", suffix);
    let js_file = format!("{}.js", artifact_name);
    let wasm_file = format!("{}.wasm", artifact_name);
    let cargo_target_dir = ctx.cargo_wasm_target_dir(use_pthreads);
    let rust_staticlib = cargo_target_dir
        .join("wasm32-unknown-emscripten")
        .join("release")
        .join("libcogentlm_wasm.a");

    println!("   -> Compiling Rust ({js_file})...");
    let rustflags = if use_pthreads {
        "-C target-feature=+atomics,+bulk-memory,+mutable-globals"
    } else {
        ""
    };
    let cargo_cmd = if cfg!(windows) {
        format!(
            "set RUSTFLAGS={rustflags}\r\ncargo build --release --package cogentlm-wasm --target wasm32-unknown-emscripten --target-dir {}",
            ctx.command_path(&cargo_target_dir)
        )
    } else if rustflags.is_empty() {
        format!(
            "cargo build --release --package cogentlm-wasm --target wasm32-unknown-emscripten --target-dir {}",
            ctx.command_path(&cargo_target_dir)
        )
    } else {
        format!(
            "RUSTFLAGS='{rustflags}' cargo build --release --package cogentlm-wasm --target wasm32-unknown-emscripten --target-dir {}",
            ctx.command_path(&cargo_target_dir)
        )
    };

    run_with_emsdk(sh, emsdk_dir, ninja_dir, &cargo_cmd)?;

    println!("   -> Linking C++ via Emscripten...");
    let wasm_dir = root.join("bindings").join("wasm");
    let wasm_source_dir = ctx.cmake_file_path(&wasm_dir);
    let rust_staticlib_cmake = ctx.cmake_file_path(&rust_staticlib);
    let build_dir = ctx.cmake_wasm_build_dir(use_pthreads);
    sh.create_dir(&build_dir)?;
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
    run_with_emsdk(sh, emsdk_dir, ninja_dir, &emcmake_cmd)?;

    let build_cmd = "cmake --build . --parallel";
    run_with_emsdk(sh, emsdk_dir, ninja_dir, build_cmd)?;
    drop(_dir);

    println!("   -> Copying artifacts to centralized NPM staging...");
    let compiled_js = build_dir.join("dist").join("CogentLM.js");
    let compiled_wasm = build_dir.join("dist").join("CogentLM.wasm");

    sh.copy_file(&compiled_js, npm_dist_wasm.join(&js_file))?;
    sh.copy_file(&compiled_wasm, npm_dist_wasm.join(&wasm_file))?;

    Ok(())
}
