//! WebAssembly/WebGPU browser build target.

use crate::cli::{WasmRuntime, WasmThreading};
use crate::javascript;
use crate::output;
use crate::toolchains::bun::setup_bun;
use crate::toolchains::cmake::setup_cmake;
use crate::toolchains::emsdk::{run_with_emsdk, setup_emsdk};
use crate::toolchains::ninja::setup_ninja;
use crate::toolchains::source::ensure_llama_cpp_submodule;
use crate::utils::BuildContext;
use anyhow::Result;
use std::path::Path;
use std::time::Instant;
use xshell::{cmd, Shell};

/// Builds the browser WASM artifacts and TypeScript package wrappers.
pub fn build(
    sh: &Shell,
    ctx: &BuildContext,
    threading: WasmThreading,
    runtime: WasmRuntime,
) -> Result<()> {
    let started_at = Instant::now();
    let root = ctx.workspace_root();
    output::phase("Browser WASM/WebGPU package");
    output::path("Workspace", root);
    output::path("WASM artifact directory", &ctx.npm_browser_wasm_dir());

    ensure_llama_cpp_submodule(sh, ctx)?;
    let emsdk_dir = setup_emsdk(sh, ctx)?;
    let cmake_bin_dir = setup_cmake(sh, ctx)?;
    let ninja_dir = setup_ninja(sh, ctx)?;

    let npm_dist_wasm = ctx.npm_browser_wasm_dir();
    if npm_dist_wasm.exists() {
        sh.remove_path(&npm_dist_wasm)?;
    }
    sh.create_dir(&npm_dist_wasm)?;

    for runtime_flavor in runtime_flavors(runtime) {
        let include_single_thread = threading.includes_single_thread()
            && (runtime_flavor.enable_webgpu || matches!(runtime, WasmRuntime::CpuNoJspi));
        if include_single_thread {
            let phase = format!("WASM {} single-thread build", runtime_flavor.label);
            output::phase(&phase);
            build_target(
                sh,
                ctx,
                root,
                &emsdk_dir,
                &cmake_bin_dir,
                ninja_dir.as_deref(),
                false,
                runtime_flavor,
                &npm_dist_wasm,
            )?;
        }

        if threading.includes_pthread() {
            let phase = format!("WASM {} pthread build", runtime_flavor.label);
            output::phase(&phase);
            build_target(
                sh,
                ctx,
                root,
                &emsdk_dir,
                &cmake_bin_dir,
                ninja_dir.as_deref(),
                true,
                runtime_flavor,
                &npm_dist_wasm,
            )?;
        }
    }

    output::phase("TypeScript browser package");
    let npm_workspace = ctx.browser_package_dir();
    let bun_exe = setup_bun(sh, ctx)?;
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
        cmd!(sh, "{bun_exe} run build:ts"),
    )?;
    output::run_build_command(
        "Staging browser package",
        cmd!(sh, "{bun_exe} run build:stage"),
    )?;

    output::success(format!(
        "WASM pipeline complete in {}",
        output::elapsed(started_at.elapsed())
    ));

    Ok(())
}

#[derive(Clone, Copy)]
struct WasmRuntimeFlavor {
    label: &'static str,
    artifact_suffix: &'static str,
    build_tag: &'static str,
    enable_webgpu: bool,
    use_jspi: bool,
}

const WEBGPU_JSPI: WasmRuntimeFlavor = WasmRuntimeFlavor {
    label: "WebGPU+JSPI",
    artifact_suffix: "",
    build_tag: "",
    enable_webgpu: true,
    use_jspi: true,
};

const CPU_NO_JSPI: WasmRuntimeFlavor = WasmRuntimeFlavor {
    label: "CPU-only non-JSPI",
    artifact_suffix: "-cpu-nojspi",
    build_tag: "cpu-nojspi",
    enable_webgpu: false,
    use_jspi: false,
};

fn runtime_flavors(runtime: WasmRuntime) -> Vec<WasmRuntimeFlavor> {
    let mut flavors = Vec::new();
    if runtime.includes_webgpu_jspi() {
        flavors.push(WEBGPU_JSPI);
    }
    if runtime.includes_cpu_nojspi() {
        flavors.push(CPU_NO_JSPI);
    }
    flavors
}

fn build_target(
    sh: &Shell,
    ctx: &BuildContext,
    root: &Path,
    emsdk_dir: &Path,
    cmake_bin_dir: &Path,
    ninja_dir: Option<&Path>,
    use_pthreads: bool,
    runtime_flavor: WasmRuntimeFlavor,
    npm_dist_wasm: &Path,
) -> Result<()> {
    let _root_dir = sh.push_dir(root);

    let threading_suffix = if use_pthreads { "-pthread" } else { "" };
    let artifact_name = format!(
        "sipp-wasm{}{}",
        threading_suffix, runtime_flavor.artifact_suffix
    );
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
        Some(cmake_bin_dir),
        ninja_dir,
        &format!("Compiling Rust staticlib for {js_file}"),
        &cargo_cmd,
    )?;

    output::step("Linking browser runtime via Emscripten");
    let wasm_dir = root.join("bindings").join("wasm");
    let wasm_source_dir = ctx.cmake_file_path(&wasm_dir);
    let rust_staticlib_cmake = ctx.cmake_file_path(&rust_staticlib);
    let build_dir = cmake_build_dir(ctx, use_pthreads, runtime_flavor);
    sh.create_dir(&build_dir)?;
    output::path("CMake build directory", &build_dir);

    let _dir = sh.push_dir(&build_dir);

    let cmake_thread_flag = if use_pthreads {
        "-DCE_USE_PTHREADS=ON"
    } else {
        "-DCE_USE_PTHREADS=OFF"
    };
    let cmake_webgpu_flag = if runtime_flavor.enable_webgpu {
        "-DCE_WASM_ENABLE_WEBGPU=ON"
    } else {
        "-DCE_WASM_ENABLE_WEBGPU=OFF"
    };
    let cmake_jspi_flag = if runtime_flavor.use_jspi {
        "-DCE_WASM_USE_JSPI=ON"
    } else {
        "-DCE_WASM_USE_JSPI=OFF"
    };
    let emcmake_cmd = format!(
        "emcmake cmake \"{}\" -G Ninja -DCMAKE_BUILD_TYPE=Release {} {} {} -DCE_WASM_RUST_STATICLIB=\"{}\"",
        wasm_source_dir,
        cmake_thread_flag,
        cmake_webgpu_flag,
        cmake_jspi_flag,
        rust_staticlib_cmake
    );
    run_with_emsdk(
        sh,
        emsdk_dir,
        Some(cmake_bin_dir),
        ninja_dir,
        &format!("Configuring CMake for {artifact_name}"),
        &emcmake_cmd,
    )?;

    let build_cmd = "cmake --build . --parallel";
    run_with_emsdk(
        sh,
        emsdk_dir,
        Some(cmake_bin_dir),
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

fn cmake_build_dir(
    ctx: &BuildContext,
    use_pthreads: bool,
    runtime_flavor: WasmRuntimeFlavor,
) -> std::path::PathBuf {
    if runtime_flavor.build_tag.is_empty() {
        return ctx.cmake_wasm_build_dir(use_pthreads);
    }

    ctx.build_root().join("cmake").join("wasm").join(format!(
        "{}-{}",
        BuildContext::wasm_build_tag(use_pthreads),
        runtime_flavor.build_tag
    ))
}
