//! Source dependency bootstrapping.

use crate::output;
use crate::utils::BuildContext;
use anyhow::Result;
use xshell::{cmd, Shell};

/// Ensures the vendored llama.cpp submodule is checked out.
pub(crate) fn ensure_llama_cpp_submodule(sh: &Shell, ctx: &BuildContext) -> Result<()> {
    let llama_dir = ctx.llama_cpp_dir();
    if llama_dir.join("CMakeLists.txt").exists() {
        output::success(format!("Using llama.cpp at {}", llama_dir.display()));
        return Ok(());
    }

    output::phase("Source dependencies");
    output::path("llama.cpp", &llama_dir);
    let _root_dir = sh.push_dir(ctx.workspace_root());
    output::run_command(
        "Initializing llama.cpp submodule",
        cmd!(
            sh,
            "git submodule update --init --recursive crates/sys/llama.cpp"
        ),
    )
}
