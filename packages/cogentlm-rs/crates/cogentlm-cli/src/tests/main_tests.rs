//! Unit tests for the parent module.

use clap::Parser;

use super::Args;

#[test]
fn gpu_layers_accepts_negative_llama_all_layers_value() {
    let args = Args::parse_from([
        "cogentlm",
        "model.gguf",
        "prompt",
        "--chat",
        "--gpu-layers",
        "-1",
    ]);

    assert_eq!(args.gpu_layers, -1);
}

#[test]
fn stats_accepts_basic_mode() {
    let args = Args::parse_from(["cogentlm", "model.gguf", "prompt", "--stats", "basic"]);

    assert_eq!(args.stats, super::CliStatsMode::Basic);
}

#[test]
fn stats_accepts_debug_mode() {
    let args = Args::parse_from(["cogentlm", "model.gguf", "prompt", "--stats", "debug"]);

    assert_eq!(args.stats, super::CliStatsMode::Debug);
}
