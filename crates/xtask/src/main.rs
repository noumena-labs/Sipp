use anyhow::Result;
use clap::{Parser, Subcommand};
use std::env;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

// Dependency versions (TODO: Move into config file)
const EMSDK_VERSION: &str = "4.0.23";
const NINJA_VERSION: &str = "1.13.2";

#[derive(Parser)]
#[command(name = "xtask", about = "CogentLM Build Orchestrator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build all targets (wasm, node, python)
    BuildAll,
    /// Build the core native Rust crates
    BuildCore,
    /// Build the WASM/WebGPU bindings using Emscripten
    BuildWasm,
    /// Build Python bindings
    BuildPython,
    /// Build Node bindings
    BuildNode,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // xshell automatically starts in the workspace root when run via `cargo xtask`
    let sh = Shell::new()?;

    match cli.command {
        Commands::BuildAll => build_all(&sh),
        Commands::BuildCore => build_core(&sh),
        Commands::BuildWasm => build_wasm(&sh),
        Commands::BuildPython => build_python(&sh),
        Commands::BuildNode => build_node(&sh),
    }
}

// workspace helper
fn workspace_root() -> PathBuf {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()));
    manifest_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

// ====================================================================================
// Build All
// ====================================================================================
fn build_all(sh: &Shell) -> Result<()> {
    build_core(sh)?;
    build_wasm(sh)?;
    build_python(sh)?;
    build_node(sh)?;
    Ok(())
}

// ====================================================================================
// Build Core
// ====================================================================================
fn build_core(sh: &Shell) -> Result<()> {
    println!("=> Building Native Rust Workspace...");
    let _dir = sh.push_dir(workspace_root());
    cmd!(sh, "cargo build --release --workspace --exclude xtask").run()?;
    Ok(())
}

// ====================================================================================
// Build Python
// ====================================================================================
fn build_python(sh: &Shell) -> Result<()> {
    println!("=> Building Python Bindings...");
    let _dir = sh.push_dir(workspace_root().join("bindings").join("python"));
    cmd!(sh, "maturin develop --release").run()?;
    Ok(())
}

// ====================================================================================
// Build Node
// ====================================================================================
fn build_node(sh: &Shell) -> Result<()> {
    println!("=> Building Node Bindings...");
    let _dir = sh.push_dir(workspace_root().join("bindings").join("node"));
    cmd!(sh, "bun install").run()?;
    cmd!(sh, "bun run build").run()?;
    Ok(())
}

// ====================================================================================
// Build WASM
// ====================================================================================
fn build_wasm(sh: &Shell) -> Result<()> {
    let root = workspace_root();
    let emsdk_dir = setup_emsdk(sh)?;
    let ninja_dir = setup_ninja(sh)?;

    // Create the final output directories
    let npm_src_wasm = root.join("packages").join("npm").join("src").join("wasm");
    let npm_dist_wasm = root.join("packages").join("npm").join("dist").join("wasm");
    sh.create_dir(&npm_src_wasm)?;
    sh.create_dir(&npm_dist_wasm)?;

    // Pass 1: Build Single-Threaded
    println!("=> Starting Phase 1: Single-Threaded Build");
    build_wasm_target(
        sh,
        &root,
        &emsdk_dir,
        ninja_dir.as_deref(),
        false,
        &npm_src_wasm,
        &npm_dist_wasm,
    )?;

    // Pass 2: Build Multi-Threaded (Pthreads)
    println!("=> Starting Phase 2: PThread Build");
    build_wasm_target(
        sh,
        &root,
        &emsdk_dir,
        ninja_dir.as_deref(),
        true,
        &npm_src_wasm,
        &npm_dist_wasm,
    )?;

    println!("=> WASM Pipeline Complete!");
    Ok(())
}

fn build_wasm_target(
    sh: &Shell,
    root: &Path,
    emsdk_dir: &Path,
    ninja_dir: Option<&Path>,
    use_pthreads: bool,
    npm_src_wasm: &Path,
    npm_dist_wasm: &Path,
) -> Result<()> {
    let _root_dir = sh.push_dir(root);

    let suffix = if use_pthreads { "-pthread" } else { "" };
    let artifact_name = format!("cogentlm-wasm{}", suffix);
    let js_file = format!("{}.js", artifact_name);
    let wasm_file = format!("{}.wasm", artifact_name);

    // 1. Compile Rust (with isolated target directories to prevent cache thrashing)
    println!("   -> Compiling Rust ({js_file})...");

    let (cargo_cmd, staticlib_path) = if use_pthreads {
        let cmd = if cfg!(windows) {
            "set RUSTFLAGS=-C target-feature=+atomics,+bulk-memory,+mutable-globals\r\ncargo build --release --package cogentlm-wasm --target wasm32-unknown-emscripten --target-dir target/pthread"
        } else {
            "RUSTFLAGS='-C target-feature=+atomics,+bulk-memory,+mutable-globals' cargo build --release --package cogentlm-wasm --target wasm32-unknown-emscripten --target-dir target/pthread"
        };
        (
            cmd,
            "../../../target/pthread/wasm32-unknown-emscripten/release/libcogentlm_wasm.a",
        )
    } else {
        let cmd = if cfg!(windows) {
            "set RUSTFLAGS=\r\ncargo build --release --package cogentlm-wasm --target wasm32-unknown-emscripten"
        } else {
            "cargo build --release --package cogentlm-wasm --target wasm32-unknown-emscripten"
        };
        (
            cmd,
            "../../../target/wasm32-unknown-emscripten/release/libcogentlm_wasm.a",
        )
    };

    run_with_emsdk(sh, emsdk_dir, ninja_dir, cargo_cmd)?;

    // 2. Compile and Link C++
    println!("   -> Linking C++ via Emscripten...");
    let wasm_dir = root.join("bindings").join("wasm");
    let build_dir = wasm_dir.join(format!("build{}", suffix));
    sh.create_dir(&build_dir)?;
    let _dir = sh.push_dir(&build_dir);

    let cmake_thread_flag = if use_pthreads {
        "-DCE_USE_PTHREADS=ON"
    } else {
        "-DCE_USE_PTHREADS=OFF"
    };
    let emcmake_cmd = format!(
        "emcmake cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release {} -DCE_WASM_RUST_STATICLIB={}",
        cmake_thread_flag, staticlib_path
    );
    run_with_emsdk(sh, emsdk_dir, ninja_dir, &emcmake_cmd)?;

    let build_cmd = "cmake --build . --parallel";
    run_with_emsdk(sh, emsdk_dir, ninja_dir, build_cmd)?;
    drop(_dir);

    // 3. Copy Artifacts to NPM Package
    println!("   -> Copying artifacts to NPM workspace...");
    let compiled_js = build_dir.join("dist").join("CogentLM.js");
    let compiled_wasm = build_dir.join("dist").join("CogentLM.wasm");

    sh.copy_file(&compiled_js, npm_src_wasm.join(&js_file))?;
    sh.copy_file(&compiled_wasm, npm_src_wasm.join(&wasm_file))?;
    sh.copy_file(&compiled_js, npm_dist_wasm.join(&js_file))?;
    sh.copy_file(&compiled_wasm, npm_dist_wasm.join(&wasm_file))?;

    Ok(())
}

fn setup_ninja(sh: &Shell) -> Result<Option<PathBuf>> {
    if cfg!(windows) {
        let root = workspace_root();
        let ninja_dir = root.join(".toolchain").join("ninja");
        let ninja_exe = ninja_dir.join("ninja.exe");

        if !ninja_exe.exists() {
            println!("=> Bootstrapping hermetic Ninja build system for Windows...");
            sh.create_dir(&ninja_dir)?;

            let url = format!(
                "https://github.com/ninja-build/ninja/releases/download/v{}/ninja-win.zip",
                NINJA_VERSION
            );
            let zip_path = ninja_dir.join("ninja-win.zip");

            // curl and tar are built into Windows 10/11 natively
            cmd!(sh, "curl -L -o {zip_path} {url}").run()?;
            cmd!(sh, "tar -xf {zip_path} -C {ninja_dir}").run()?;
            sh.remove_path(zip_path)?;
        }
        Ok(Some(ninja_dir))
    } else {
        Ok(None)
    }
}

fn setup_emsdk(sh: &Shell) -> Result<PathBuf> {
    let root = workspace_root();
    let emsdk_dir = root.join(".toolchain").join("emsdk");

    if !emsdk_dir.exists() {
        println!("=> Bootstrapping hermetic Emscripten toolchain...");
        println!("   Location: {}", emsdk_dir.display());

        let _root_dir = sh.push_dir(&root);
        cmd!(sh, "git clone https://github.com/emscripten-core/emsdk.git")
            .arg(&emsdk_dir)
            .run()?;
    }

    let _dir = sh.push_dir(&emsdk_dir);
    println!("=> Installing and Activating emsdk v{EMSDK_VERSION}...");
    if cfg!(windows) {
        cmd!(sh, "cmd.exe /c emsdk.bat install {EMSDK_VERSION}").run()?;
        cmd!(sh, "cmd.exe /c emsdk.bat activate {EMSDK_VERSION}").run()?;
    } else {
        cmd!(sh, "bash -c")
            .arg(format!("./emsdk install {EMSDK_VERSION}"))
            .run()?;
        cmd!(sh, "bash -c")
            .arg(format!("./emsdk activate {EMSDK_VERSION}"))
            .run()?;
    }

    Ok(emsdk_dir)
}

fn run_with_emsdk(
    sh: &Shell,
    emsdk_dir: &Path,
    ninja_dir: Option<&Path>,
    command: &str,
) -> Result<()> {
    if cfg!(windows) {
        // Hack to bypass cmd.exe nested quoting hell
        let bat = emsdk_dir.join("emsdk_env.bat");
        let temp_script = sh.current_dir().join(".run_emsdk_wrapper.bat");

        // Inject Ninja into the PATH if it was provided
        let path_injection = if let Some(n_dir) = ninja_dir {
            format!("set PATH={};%PATH%\r\n", n_dir.display())
        } else {
            String::new()
        };

        let script_content = format!(
            "@echo off\r\n\
            call \"{}\"\r\n\
            {}\
            set EMCMAKE=emcmake.bat\r\n\
            set EMMAKE=emmake.bat\r\n\
            {}\r\n",
            bat.display(),
            path_injection,
            command
        );
        sh.write_file(&temp_script, script_content)?;
        let res = cmd!(sh, "cmd.exe /c {temp_script}").run();

        let _ = sh.remove_path(&temp_script);
        res?;
    } else {
        let script = emsdk_dir.join("emsdk_env.sh").display().to_string();
        let full_cmd = format!("source \"{}\" && {}", script, command);
        cmd!(sh, "bash -c").arg(full_cmd).run()?;
    }
    Ok(())
}
