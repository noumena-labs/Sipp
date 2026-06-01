//! Shared sample GGUF model cache used by setup and smoke tests.

use crate::output;
use crate::utils::BuildContext;
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use xshell::{cmd, Shell};

const SAMPLE_MODEL_URL: &str =
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf";
const SAMPLE_MODEL_FILE: &str = "qwen2.5-0.5b-instruct-q4_0.gguf";
const SAMPLE_MODEL_SHA256: &str =
    "7671c0c304e6ce5a7fc577bcb12aba01e2c155cc2efd29b2213c95b18edaf6ed";

/// Controls how the sample model cache is resolved.
pub struct SampleModelOptions {
    /// Whether the model can be downloaded when missing or invalid.
    pub allow_download: bool,
}

/// Returns the source URL for the pinned sample model.
pub fn sample_model_url() -> &'static str {
    SAMPLE_MODEL_URL
}

/// Returns the canonical cached sample model path.
pub fn sample_model_path(ctx: &BuildContext) -> PathBuf {
    ctx.sample_models_dir().join(SAMPLE_MODEL_FILE)
}

/// Returns a displayable sample model argument placeholder.
pub fn sample_model_arg(ctx: &BuildContext) -> String {
    let model_path = sample_model_path(ctx);
    if model_path.exists() {
        model_path.display().to_string()
    } else {
        "<model.gguf>".to_owned()
    }
}

/// Ensures the cached sample GGUF model exists and matches the pinned hash.
pub fn ensure_sample_model(
    sh: &Shell,
    ctx: &BuildContext,
    options: SampleModelOptions,
) -> Result<PathBuf> {
    output::phase("Sample model");
    let model_dir = ctx.sample_models_dir();
    sh.create_dir(&model_dir)
        .with_context(|| format!("failed to create {}", model_dir.display()))?;

    let model_path = sample_model_path(ctx);
    if model_path.exists() {
        match validate_sample_model(&model_path) {
            Ok(()) => {
                output::success(format!(
                    "Using verified sample model at {}",
                    model_path.display()
                ));
                return Ok(model_path);
            }
            Err(error) if options.allow_download => {
                output::warning(format!(
                    "Cached sample model failed validation and will be replaced: {error:#}"
                ));
                sh.remove_path(&model_path)?;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "cached sample model failed validation in offline mode: {}",
                        model_path.display()
                    )
                });
            }
        }
    }

    if !options.allow_download {
        bail!(
            "sample model is missing at {} and downloads are disabled",
            model_path.display()
        );
    }

    output::detail("Download", SAMPLE_MODEL_URL);
    output::path("Destination", &model_path);
    output::run_command(
        "Downloading sample GGUF model",
        cmd!(sh, "curl -f -L -o {model_path} {SAMPLE_MODEL_URL}"),
    )?;
    validate_sample_model(&model_path)?;
    Ok(model_path)
}

fn validate_sample_model(path: &Path) -> Result<()> {
    let actual = sha256_file(path)?;
    if actual != SAMPLE_MODEL_SHA256 {
        bail!(
            "sample model checksum mismatch for {}; expected {}, got {}",
            path.display(),
            SAMPLE_MODEL_SHA256,
            actual
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let bytes = file
            .read(&mut buffer)
            .with_context(|| format!("failed to read {}", path.display()))?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}
