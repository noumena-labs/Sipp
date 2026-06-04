//! Tests the `sample_model` module in `xtask`.
//!
//! Covers sample model path formatting, checksum validation, and offline error
//! paths with fake cache directories instead of downloading or executing model
//! inference.

use sha2::{Digest, Sha256};

use crate::test_support::TempDir;
use crate::utils::BuildContext;

use super::{
    ensure_sample_model, sample_model_arg, sample_model_path, sample_model_url, sha256_file,
    validate_sample_model, SampleModelOptions,
};

#[test]
fn sample_model_paths_and_display_arg_use_fake_workspace_cache() {
    let temp = TempDir::new("sample-paths");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let path = sample_model_path(&ctx);

    assert_eq!(
        sample_model_url(),
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_0.gguf"
    );
    assert!(path.ends_with("qwen2.5-0.5b-instruct-q4_0.gguf"));
    assert_eq!(sample_model_arg(&ctx), "<model.gguf>");

    temp.write(
        ".build/models/qwen2.5-0.5b-instruct-q4_0.gguf",
        "not a real model",
    );
    assert_eq!(sample_model_arg(&ctx), path.display().to_string());
}

#[test]
fn checksum_helper_hashes_file_contents() {
    let temp = TempDir::new("sample-hash");
    let path = temp.write("model.gguf", "hello");
    let expected = format!("{:x}", Sha256::digest(b"hello"));

    assert_eq!(sha256_file(&path).unwrap(), expected);
}

#[test]
fn validation_reports_mismatched_cached_model() {
    let temp = TempDir::new("sample-validate");
    let path = temp.write("model.gguf", "bad model");

    let error = validate_sample_model(&path).unwrap_err();
    assert!(format!("{error:#}").contains("checksum mismatch"));
}

#[test]
fn offline_missing_model_errors_without_download() {
    let temp = TempDir::new("sample-offline-missing");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let sh = xshell::Shell::new().unwrap();

    let error = ensure_sample_model(
        &sh,
        &ctx,
        SampleModelOptions {
            allow_download: false,
        },
    )
    .unwrap_err();

    assert!(format!("{error:#}").contains("downloads are disabled"));
}

#[test]
fn offline_invalid_cached_model_preserves_file_and_reports_validation_context() {
    let temp = TempDir::new("sample-offline-invalid");
    let ctx = BuildContext::from_workspace_root_for_test(temp.path());
    let model = temp.write(".build/models/qwen2.5-0.5b-instruct-q4_0.gguf", "bad model");
    let sh = xshell::Shell::new().unwrap();

    let error = ensure_sample_model(
        &sh,
        &ctx,
        SampleModelOptions {
            allow_download: false,
        },
    )
    .unwrap_err();

    assert!(model.exists());
    assert!(format!("{error:#}").contains("offline mode"));
}
