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

    println!("=> Step 1: Compiling Rust core to WASM staticlib...");
    let _root_dir = sh.push_dir(&root);
    run_with_emsdk(
        sh,
        &emsdk_dir,
        ninja_dir.as_deref(),
        "cargo build --release --package cogentlm-wasm --target wasm32-unknown-emscripten",
    )?;
    drop(_root_dir);

    println!("=> Step 2: Linking C++ and Rust via Emscripten...");
    let wasm_dir = root.join("bindings").join("wasm");
    let build_dir = wasm_dir.join("build");
    sh.create_dir(&build_dir)?;

    let _dir = sh.push_dir(&build_dir);

    // 1. ADD '-G Ninja' to the CMake configuration command
    let emcmake_cmd = "emcmake cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release -DCE_WASM_RUST_STATICLIB=../../../target/wasm32-unknown-emscripten/release/libcogentlm_wasm.a";
    run_with_emsdk(sh, &emsdk_dir, ninja_dir.as_deref(), emcmake_cmd)?;

    // 2. REPLACE `emmake make -j` with CMake's universal build command
    let build_cmd = "cmake --build . --parallel";
    run_with_emsdk(sh, &emsdk_dir, ninja_dir.as_deref(), build_cmd)?;

    drop(_dir);

    println!("=> Step 3: Copying WASM artifacts to NPM package...");
    let npm_native_dir = root.join("packages").join("npm").join("src").join("native");
    sh.create_dir(&npm_native_dir)?;

    sh.copy_file(build_dir.join("dist").join("CogentLM.js"), &npm_native_dir)?;
    sh.copy_file(
        build_dir.join("dist").join("CogentLM.wasm"),
        &npm_native_dir,
    )?;

    println!("=> WASM Pipeline Complete!");
    Ok(())
}

fn setup_ninja(sh: &Shell) -> Result<Option<PathBuf>> {
    if cfg!(windows) {
        let root = workspace_root();
        let ninja_dir = root.join("third_party").join("ninja");
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
    let emsdk_dir = root.join("third_party").join("emsdk");

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
