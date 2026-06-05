//! CUDA toolkit discovery.

use crate::output;
use anyhow::Result;
use std::env;
use std::path::PathBuf;

/////////////////////////////////////////////////////////////////////////////////
/// TESTS
/////////////////////////////////////////////////////////////////////////////////

#[cfg(test)]
#[path = "../tests/toolchains/cuda_tests.rs"]
mod cuda_tests;

/////////////////////////////////////////////////////////////////////////////////
/// SRC
/////////////////////////////////////////////////////////////////////////////////

/// Finds a CUDA toolkit installation suitable for CMake/NVCC builds.
pub(crate) fn setup_cuda() -> Result<PathBuf> {
    output::phase("CUDA Toolkit");

    let cuda_env = env::var_os("CUDA_PATH").or_else(|| env::var_os("CUDA_HOME"));

    if let Some(path) = cuda_env {
        let cuda_path = PathBuf::from(path);
        let nvcc_exe = if cfg!(windows) {
            cuda_path.join("bin").join("nvcc.exe")
        } else {
            cuda_path.join("bin").join("nvcc")
        };

        if nvcc_exe.exists() {
            output::success(format!("Using CUDA Toolkit at {}", cuda_path.display()));
            return Ok(cuda_path);
        }
    }

    output::warning("CUDA Toolkit not found");
    output::detail("Required for", "CUDA backend builds");
    output::detail("Download", "https://developer.nvidia.com/cuda-downloads");
    output::detail("Expected env", "CUDA_PATH or CUDA_HOME pointing at nvcc");
    output::detail("Retry", "cargo xtask build node --backend cuda");
    anyhow::bail!("Missing NVIDIA CUDA Toolkit.");
}
