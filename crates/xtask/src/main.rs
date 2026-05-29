use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use std::env;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

// Dependency versions (TODO: Move into config file)
const EMSDK_VERSION: &str = "4.0.23";
const NINJA_VERSION: &str = "1.13.2";
const VULKAN_VERSION: &str = "1.4.350.0";

#[derive(Parser)]
#[command(name = "xtask", about = "CogentLM Build Orchestrator")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

const NODE_BINARY_NAME: &str = "cogentlm_node";

#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Backend {
    /// Standard CPU computation fallback
    Cpu,
    /// NVIDIA CUDA hardware acceleration
    Cuda,
    /// Apple Metal native acceleration
    Metal,
    /// Vulkan cross-platform GPU acceleration
    Vulkan,
    /// Build all supported Node backends for the host OS
    All,
}

impl Backend {
    /// Helper to convert the enum into lowercase string for CLI features
    fn as_str(&self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
            Backend::All => "all",
        }
    }
}

#[derive(Subcommand)]
enum Commands {
    /// Build all targets (wasm, node, python)
    BuildAll,
    /// Build the core native Rust crates
    BuildCore,
    /// Build the WASM/WebGPU bindings using Emscripten
    BuildWasm,
    /// Build Python bindings [BACKEND] = {cpu | cuda | metal | vulkan}
    BuildPython {
        /// The computation backend to compile against
        #[arg(long, short)]
        backend: Option<Backend>,
    },
    /// Build Node bindings [BACKEND] = {cpu | cuda | metal | vulkan}
    BuildNode {
        /// The computation backend to compile against
        #[arg(long, short)]
        backend: Option<Backend>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // xshell automatically starts in the workspace root when run via `cargo xtask`
    let sh = Shell::new()?;

    match cli.command {
        Commands::BuildAll => build_all(&sh),
        Commands::BuildCore => build_core(&sh),
        Commands::BuildWasm => build_wasm(&sh),
        Commands::BuildPython { backend } => build_python(&sh, backend.as_ref()),
        Commands::BuildNode { backend } => build_node(&sh, backend.as_ref()),
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
    build_python(sh, None)?;
    build_node(sh, None)?;
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
fn build_python(sh: &Shell, backend: Option<&Backend>) -> Result<()> {
    if matches!(backend, Some(Backend::All)) {
        anyhow::bail!("--backend all is only supported for Node bindings");
    }

    println!("=> Building Python Bindings...");
    let _dir = sh.push_dir(workspace_root().join("bindings").join("python"));

    let mut maturin_cmd = cmd!(sh, "uv run maturin develop --release");
    maturin_cmd = apply_toolchains(sh, maturin_cmd, backend)?;

    match backend {
        Some(Backend::Cpu) | None => {
            println!("   Hardware Backend: CPU (Default)");
        }
        Some(b) => {
            let feature = b.as_str();
            println!("   Hardware Backend: {}", feature.to_uppercase());
            maturin_cmd = maturin_cmd.arg("--features").arg(feature);
        }
    }

    maturin_cmd.run()?;
    Ok(())
}

// ====================================================================================
// Build Node
// ====================================================================================
fn build_node(sh: &Shell, backend: Option<&Backend>) -> Result<()> {
    println!("=> Building Node Bindings...");
    let node_dir = workspace_root().join("bindings").join("node");
    let _dir = sh.push_dir(&node_dir);

    cmd!(sh, "bun install").run()?;

    let dist_dir = node_dir.join("dist");
    prepare_node_dist_dir(sh, &dist_dir)?;

    let best_effort = matches!(backend, Some(Backend::All));
    let backends_to_build = node_backends_to_build(backend);
    let mut built = Vec::new();
    let mut skipped = Vec::new();

    for backend in backends_to_build {
        let optional = best_effort && backend != Backend::Cpu;
        match build_node_backend_variant(sh, &dist_dir, &backend) {
            Ok(path) => {
                println!("   Wrote {}", path.display());
                built.push(backend);
            }
            Err(error) if optional => {
                eprintln!(
                    "   Warning: skipped optional {} backend: {error:#}",
                    backend.as_str()
                );
                skipped.push(backend);
            }
            Err(error) => return Err(error),
        }
    }

    let built_names = built
        .iter()
        .map(Backend::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    println!("=> Node Build Complete! Built variants: {built_names}");

    if !skipped.is_empty() {
        let skipped_names = skipped
            .iter()
            .map(Backend::as_str)
            .collect::<Vec<_>>()
            .join(", ");
        println!("   Optional variants skipped: {skipped_names}");
    }

    Ok(())
}

fn node_backends_to_build(backend: Option<&Backend>) -> Vec<Backend> {
    match backend {
        Some(Backend::All) => {
            if cfg!(target_os = "macos") {
                vec![Backend::Cpu, Backend::Metal]
            } else {
                vec![Backend::Cpu, Backend::Vulkan, Backend::Cuda]
            }
        }
        Some(backend) => vec![backend.clone()],
        None => vec![Backend::Cpu],
    }
}

fn prepare_node_dist_dir(sh: &Shell, dist_dir: &Path) -> Result<()> {
    sh.create_dir(dist_dir)?;

    let staging_dir = dist_dir.join(".staging");
    if staging_dir.exists() {
        sh.remove_path(&staging_dir)?;
    }

    for entry in std::fs::read_dir(dist_dir)
        .with_context(|| format!("failed to read {}", dist_dir.display()))?
    {
        let path = entry?.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with(&format!("{NODE_BINARY_NAME}_"))
            && path.extension().and_then(|ext| ext.to_str()) == Some("node")
        {
            sh.remove_path(path)?;
        }
    }

    Ok(())
}

fn build_node_backend_variant(
    sh: &Shell,
    dist_dir: &Path,
    backend: &Backend,
) -> Result<PathBuf> {
    if matches!(backend, Backend::All) {
        anyhow::bail!("Backend::All cannot be built as a single Node variant");
    }

    let feature = backend.as_str();
    println!("--------------------------------------------------");
    println!("Compiling Node Variant: {}", feature.to_uppercase());
    println!("--------------------------------------------------");

    let staging_dir = dist_dir.join(".staging").join(feature);
    if staging_dir.exists() {
        sh.remove_path(&staging_dir)?;
    }
    sh.create_dir(&staging_dir)?;

    let target_dir = workspace_root().join("target").join("node").join(feature);
    let mut napi_cmd = cmd!(
        sh,
        "bunx napi build --platform --release --no-js --output-dir {staging_dir} --target-dir {target_dir}"
    );
    napi_cmd = apply_toolchains(sh, napi_cmd, Some(backend))?;

    if *backend != Backend::Cpu {
        napi_cmd = napi_cmd.arg("--features").arg(feature);
    }

    napi_cmd
        .run()
        .with_context(|| format!("failed to build Node {feature} backend"))?;

    let artifact = find_node_artifact(&staging_dir)?.with_context(|| {
        format!(
            "napi did not produce a .node artifact in {}",
            staging_dir.display()
        )
    })?;
    let file_name = artifact
        .file_name()
        .and_then(|name| name.to_str())
        .with_context(|| format!("invalid Node artifact path {}", artifact.display()))?;
    let renamed = file_name.replacen(
        NODE_BINARY_NAME,
        &format!("{NODE_BINARY_NAME}_{feature}"),
        1,
    );
    if renamed == file_name {
        anyhow::bail!("unexpected Node artifact name: {file_name}");
    }

    let dest = dist_dir.join(renamed);
    sh.copy_file(&artifact, &dest)?;
    sh.remove_path(&staging_dir)?;

    Ok(dest)
}

fn find_node_artifact(dir: &Path) -> Result<Option<PathBuf>> {
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("failed to read {}", dir.display()))?
    {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("node") {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

// ====================================================================================
// Build WASM (Unchanged)
// ====================================================================================
fn build_wasm(sh: &Shell) -> Result<()> {
    let root = workspace_root();
    let emsdk_dir = setup_emsdk(sh)?;
    let ninja_dir = setup_ninja(sh)?;

    let npm_src_wasm = root.join("packages").join("npm").join("src").join("wasm");
    let npm_dist_wasm = root.join("packages").join("npm").join("dist").join("wasm");
    sh.create_dir(&npm_src_wasm)?;
    sh.create_dir(&npm_dist_wasm)?;

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

    println!("   -> Copying artifacts to NPM workspace...");
    let compiled_js = build_dir.join("dist").join("CogentLM.js");
    let compiled_wasm = build_dir.join("dist").join("CogentLM.wasm");

    sh.copy_file(&compiled_js, npm_src_wasm.join(&js_file))?;
    sh.copy_file(&compiled_wasm, npm_src_wasm.join(&wasm_file))?;
    sh.copy_file(&compiled_js, npm_dist_wasm.join(&js_file))?;
    sh.copy_file(&compiled_wasm, npm_dist_wasm.join(&wasm_file))?;

    Ok(())
}

// ====================================================================================
// Toolchain Bootstrappers
// ====================================================================================

fn apply_toolchains<'a>(
    sh: &Shell,
    mut cmd: xshell::Cmd<'a>,
    backend: Option<&Backend>,
) -> Result<xshell::Cmd<'a>> {
    let ninja_dir = setup_ninja(sh)?;
    let mut path_additions = Vec::new();

    // 1. Force Ninja as the generator
    if let Some(n_dir) = &ninja_dir {
        path_additions.push(n_dir.display().to_string());
        cmd = cmd.env("CMAKE_GENERATOR", "Ninja");
    }

    // 2. Handle Hardware Specifics
    match backend {
        Some(Backend::Vulkan) => {
            let vk = setup_vulkan(sh)?;
            let bin_path = if cfg!(windows) {
                vk.join("Bin")
            } else if cfg!(target_os = "macos") {
                vk.join("macOS").join("bin")
            } else {
                vk.join(VULKAN_VERSION).join("x86_64").join("bin")
            };
            path_additions.push(bin_path.display().to_string());

            let vulkan_sdk_path = if cfg!(windows) {
                vk.to_path_buf()
            } else if cfg!(target_os = "macos") {
                vk.join("macOS")
            } else {
                vk.join(VULKAN_VERSION).join("x86_64")
            };
            cmd = cmd.env("VULKAN_SDK", &vulkan_sdk_path);

            let current_cmake_prefix = env::var("CMAKE_PREFIX_PATH").unwrap_or_default();
            let separator = if cfg!(windows) { ";" } else { ":" };
            let new_cmake_prefix = if current_cmake_prefix.is_empty() {
                vulkan_sdk_path.display().to_string()
            } else {
                format!(
                    "{}{separator}{}",
                    vulkan_sdk_path.display(),
                    current_cmake_prefix
                )
            };
            cmd = cmd.env("CMAKE_PREFIX_PATH", new_cmake_prefix);
        }
        Some(Backend::Cuda) => {
            let cuda_path = setup_cuda(sh)?;
            let bin_path = cuda_path.join("bin");
            path_additions.push(bin_path.display().to_string());

            // Strictly enforce the compiler path for CMake
            let nvcc_exe = if cfg!(windows) {
                bin_path.join("nvcc.exe")
            } else {
                bin_path.join("nvcc")
            };
            cmd = cmd.env("CUDACXX", nvcc_exe.display().to_string());
            cmd = cmd.env("CUDA_TOOLKIT_ROOT_DIR", cuda_path.display().to_string());
        }
        _ => {}
    }

    // 3. Construct the final PATH
    if !path_additions.is_empty() {
        let current_path = env::var("PATH").unwrap_or_default();
        let separator = if cfg!(windows) { ";" } else { ":" };
        let new_path = format!(
            "{}{separator}{}",
            path_additions.join(separator),
            current_path
        );
        cmd = cmd.env("PATH", new_path);
    }

    Ok(cmd)
}

fn setup_cuda(_sh: &Shell) -> Result<PathBuf> {
    println!("=> Validating NVIDIA CUDA Toolkit...");

    // 1. Check standard environment variables set by the NVIDIA installer
    let cuda_env = env::var_os("CUDA_PATH").or_else(|| env::var_os("CUDA_HOME"));

    if let Some(path) = cuda_env {
        let cuda_path = PathBuf::from(path);
        let nvcc_exe = if cfg!(windows) {
            cuda_path.join("bin").join("nvcc.exe")
        } else {
            cuda_path.join("bin").join("nvcc")
        };

        if nvcc_exe.exists() {
            println!("   Found CUDA Toolkit at: {}", cuda_path.display());
            return Ok(cuda_path);
        }
    }

    // 2. If env vars are missing or nvcc isn't there, throw the DX-friendly error
    println!("====================================================================");
    println!("❌ CUDA TOOLKIT NOT FOUND ❌");
    println!("====================================================================");
    println!("");
    println!("To compile the CUDA backend, you must install the NVIDIA CUDA Toolkit.");
    println!("  1. Download the latest toolkit from NVIDIA:");
    println!("     https://developer.nvidia.com/cuda-downloads");
    println!("");
    println!("  2. Install it (Leave default settings to automatically set CUDA_PATH)");
    println!("");
    println!("  3. Restart your terminal / VS Code to reload environment variables.");
    println!("");
    println!("  4. Run this build command again.");
    println!("");
    println!("====================================================================");
    anyhow::bail!("Missing NVIDIA CUDA Toolkit.");
}

fn setup_vulkan(sh: &Shell) -> Result<PathBuf> {
    let root = workspace_root();
    let toolchain_dir = root.join(".toolchain");
    let vulkan_dir = toolchain_dir.join("vulkan");

    // Map the OS to LunarG's specific URL routing and internal folder structures
    let (os_path, filename, bin_path) = if cfg!(windows) {
        (
            "windows",
            format!("vulkansdk-windows-X64-{VULKAN_VERSION}.exe"),
            vulkan_dir.join("Bin").join("glslc.exe"),
        )
    } else if cfg!(target_os = "macos") {
        (
            "mac",
            format!("vulkansdk-macos-{VULKAN_VERSION}.zip"),
            vulkan_dir.join("macOS").join("bin").join("glslc"),
        )
    } else {
        (
            "linux",
            format!("vulkansdk-linux-x86_64-{VULKAN_VERSION}.tar.xz"),
            vulkan_dir
                .join(VULKAN_VERSION)
                .join("x86_64")
                .join("bin")
                .join("glslc"),
        )
    };

    if !bin_path.exists() {
        println!("=> Bootstrapping hermetic Vulkan SDK...");
        sh.create_dir(&vulkan_dir)?;

        let url =
            format!("https://sdk.lunarg.com/sdk/download/{VULKAN_VERSION}/{os_path}/{filename}");
        let archive_path = toolchain_dir.join(&filename);

        println!("   Downloading Vulkan SDK (~400MB) from:");
        println!("   {url}");

        // '-f' ensures it fails on 404s instead of downloading dummy HTML files
        cmd!(sh, "curl -f -L -o {archive_path} {url}").run()?;

        println!("   Extracting/Installing into .toolchain/vulkan...");
        if cfg!(windows) {
            cmd!(sh, "{archive_path} --root {vulkan_dir} --accept-licenses --default-answer --confirm-command install copy_only=1").run()?;
        } else if cfg!(target_os = "macos") {
            cmd!(sh, "unzip -q {archive_path} -d {vulkan_dir}").run()?;
        } else {
            cmd!(sh, "tar -xf {archive_path} -C {vulkan_dir}").run()?;
        }

        // Clean up the massive installer/archive
        sh.remove_path(&archive_path)?;
    }

    Ok(vulkan_dir)
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
        let _root_dir = sh.push_dir(&root);
        cmd!(sh, "git clone https://github.com/emscripten-core/emsdk.git")
            .arg(&emsdk_dir)
            .run()?;
    }

    let _dir = sh.push_dir(&emsdk_dir);
    println!("=> Activating emsdk v{EMSDK_VERSION}...");
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
        let bat = emsdk_dir.join("emsdk_env.bat");
        let temp_script = sh.current_dir().join(".run_emsdk_wrapper.bat");

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
