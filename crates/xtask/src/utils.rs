//! Shared build paths and formatting helpers.

use crate::cli::Backend;
use anyhow::{Context, Result};
use std::env;
use std::path::{Path, PathBuf};

const BROWSER_PACKAGE_ARTIFACT_DIR: &str = "cogentlm-browser";

/// Shared immutable context for xtask build paths.
#[derive(Clone, Debug)]
pub struct BuildContext {
    workspace_root: PathBuf,
}

impl BuildContext {
    /// Creates a build context rooted at the Cargo workspace.
    pub fn new() -> Result<Self> {
        let manifest_dir = PathBuf::from(
            env::var("CARGO_MANIFEST_DIR")
                .unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_owned()),
        );
        let workspace_root = manifest_dir
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf)
            .with_context(|| {
                format!(
                    "failed to resolve workspace root from {}",
                    manifest_dir.display()
                )
            })?;

        Ok(Self { workspace_root })
    }

    pub(crate) fn workspace_root(&self) -> &Path {
        &self.workspace_root
    }

    pub(crate) fn build_root(&self) -> PathBuf {
        self.workspace_root.join(".build")
    }

    pub(crate) fn cargo_build_root(&self) -> PathBuf {
        self.build_root().join("cargo")
    }

    pub(crate) fn cargo_node_target_dir(&self, backend: &Backend) -> PathBuf {
        self.cargo_build_root().join("node").join(backend.as_str())
    }

    pub(crate) fn cargo_python_target_dir(&self, backend: Option<&Backend>) -> PathBuf {
        self.cargo_build_root()
            .join("python")
            .join(Self::backend_build_tag(backend))
    }

    pub(crate) fn cargo_wasm_target_dir(&self, use_pthreads: bool) -> PathBuf {
        self.cargo_build_root()
            .join("wasm")
            .join(Self::wasm_build_tag(use_pthreads))
    }

    pub(crate) fn cmake_wasm_build_dir(&self, use_pthreads: bool) -> PathBuf {
        self.build_root()
            .join("cmake")
            .join("wasm")
            .join(Self::wasm_build_tag(use_pthreads))
    }

    pub(crate) fn artifacts_root(&self) -> PathBuf {
        self.build_root().join("artifacts")
    }

    pub(crate) fn node_artifacts_dir(&self) -> PathBuf {
        self.artifacts_root().join("node")
    }

    pub(crate) fn npm_browser_artifacts_dir(&self) -> PathBuf {
        self.artifacts_root()
            .join("npm")
            .join(BROWSER_PACKAGE_ARTIFACT_DIR)
    }

    pub(crate) fn npm_browser_wasm_dir(&self) -> PathBuf {
        self.npm_browser_artifacts_dir().join("dist").join("wasm")
    }

    pub(crate) fn toolchain_dir(&self) -> PathBuf {
        self.build_root().join("toolchain")
    }

    pub(crate) fn tmp_dir(&self) -> PathBuf {
        self.build_root().join("tmp")
    }

    pub(crate) fn backend_build_tag(backend: Option<&Backend>) -> &'static str {
        backend.map(Backend::as_str).unwrap_or("cpu")
    }

    pub(crate) fn wasm_build_tag(use_pthreads: bool) -> &'static str {
        if use_pthreads {
            "pthread"
        } else {
            "single"
        }
    }

    pub(crate) fn command_path(&self, path: &Path) -> String {
        format!("\"{}\"", path.display())
    }

    pub(crate) fn cmake_file_path(&self, path: &Path) -> String {
        path.display().to_string().replace('\\', "/")
    }
}
