//! Tests the `main` module in `sipp-cli`.
//!
//! Covers CLI parsing, configuration mapping, stats rendering, and command behavior without running model-backed inference unless marked as an external smoke test.

use clap::Parser;
use sipp::engine::GpuLayerConfig;
use sipp::lifecycle::{BackendPreference, StatsMode};
use sipp::runtime::metrics::RuntimeObservabilityMetrics;

use super::*;

#[test]
fn gpu_layers_accepts_negative_llama_all_layers_value() {
    let args = Args::parse_from([
        "sipp",
        "model.gguf",
        "prompt",
        "--chat",
        "--gpu-layers",
        "-1",
    ]);

    assert_eq!(args.gpu_layers, Some(-1));
}

#[test]
fn stats_accepts_basic_mode() {
    let args = Args::parse_from(["sipp", "model.gguf", "prompt", "--stats", "basic"]);

    assert_eq!(args.stats, super::CliStatsMode::Basic);
}

#[test]
fn invalid_backend_and_stats_modes_are_rejected_by_clap() {
    assert!(
        Args::try_parse_from(["sipp", "model.gguf", "prompt", "--backend", "bogus"]).is_err()
    );
    assert!(
        Args::try_parse_from(["sipp", "model.gguf", "prompt", "--stats", "verbose"]).is_err()
    );
}

#[test]
fn backend_and_stats_modes_map_to_lifecycle_preferences() {
    assert_eq!(CliBackend::Auto.to_preference(), BackendPreference::Auto);
    assert_eq!(CliBackend::Cpu.to_preference(), BackendPreference::Cpu);
    assert_eq!(
        CliBackend::Vulkan.to_preference(),
        BackendPreference::Vulkan
    );

    assert_eq!(CliStatsMode::Off.to_lifecycle_stats(), StatsMode::Off);
    assert_eq!(CliStatsMode::Basic.to_lifecycle_stats(), StatsMode::Basic);
    assert_eq!(
        CliStatsMode::Profile.to_lifecycle_stats(),
        StatsMode::Profile
    );
}

#[test]
fn runtime_config_from_args_clamps_sizes_and_forces_greedy_top_k() {
    let args = Args::parse_from([
        "sipp",
        "model.gguf",
        "prompt",
        "--ctx-size",
        "4294967295",
        "--batch-size",
        "4294967295",
        "--threads",
        "6",
        "--temperature",
        "0",
        "--top-k",
        "40",
        "--seed",
        "7",
        "--gpu-layers",
        "-1",
    ]);

    let config = runtime_config_from_args(&args);

    assert_eq!(config.context.n_ctx, Some(i32::MAX));
    assert_eq!(config.context.n_batch, Some(i32::MAX));
    assert_eq!(config.context.n_ubatch, Some(i32::MAX));
    assert_eq!(config.context.n_threads, Some(6));
    assert_eq!(config.context.n_threads_batch, Some(6));
    assert_eq!(config.context.n_parallel, Some(1));
    assert_eq!(config.sampling.temperature, Some(0.0));
    assert_eq!(config.sampling.top_k, Some(1));
    assert_eq!(config.sampling.seed, Some(7));
    assert_eq!(config.placement.gpu_layers, GpuLayerConfig::All);
}

#[test]
fn runtime_config_omits_default_sentinel_seed() {
    let args = Args::parse_from(["sipp", "model.gguf", "prompt"]);

    let config = runtime_config_from_args(&args);

    assert_eq!(config.sampling.seed, None);
}

#[test]
fn stats_formatting_respects_off_basic_and_profile_modes() {
    let stats = RuntimeObservabilityMetrics {
        input_tokens: 3,
        output_tokens: 2,
        prefill_tokens: 3,
        cache_hits: 1,
        ttft_ms: 10.0,
        itl_avg_ms: 5.0,
        e2e_ms: 1000.0,
        decode_ms: 500.0,
        native_gpu_ms: 2.0,
        native_sync_ms: 3.0,
        native_logic_ms: 4.0,
        ..RuntimeObservabilityMetrics::default()
    };

    let mut off = Vec::new();
    print_stats_to_writer(CliStatsMode::Off, stats, &mut off).expect("off stats");
    assert!(off.is_empty());

    let mut basic = Vec::new();
    print_stats_to_writer(CliStatsMode::Basic, stats, &mut basic).expect("basic stats");
    let basic = String::from_utf8(basic).expect("basic utf8");
    assert!(basic.contains("input_tokens: 3"));
    assert!(basic.contains("e2e_tokens_per_second: 2.00"));
    assert!(!basic.contains("backend_ms"));

    let mut profile = Vec::new();
    print_stats_to_writer(CliStatsMode::Profile, stats, &mut profile).expect("profile stats");
    let profile = String::from_utf8(profile).expect("profile utf8");
    assert!(profile.contains("backend_ms: 2.00"));
    assert!(profile.contains("sync_ms: 3.00"));
    assert!(profile.contains("engine_overhead_ms: 4.00"));
}
