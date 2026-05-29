//! CUDA toolkit discovery.

use anyhow::Result;
use std::env;
use std::path::PathBuf;

/// Finds a CUDA toolkit installation suitable for CMake/NVCC builds.
pub(crate) fn setup_cuda() -> Result<PathBuf> {
    println!("=> Validating NVIDIA CUDA Toolkit...");

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

    println!("====================================================================");
    println!("CUDA TOOLKIT NOT FOUND");
    println!("====================================================================");
    println!();
    println!("To compile the CUDA backend, you must install the NVIDIA CUDA Toolkit.");
    println!("  1. Download the latest toolkit from NVIDIA:");
    println!("     https://developer.nvidia.com/cuda-downloads");
    println!();
    println!("  2. Install it (Leave default settings to automatically set CUDA_PATH)");
    println!();
    println!("  3. Restart your terminal / VS Code to reload environment variables.");
    println!();
    println!("  4. Run this build command again.");
    println!();
    println!("====================================================================");
    anyhow::bail!("Missing NVIDIA CUDA Toolkit.");
}
