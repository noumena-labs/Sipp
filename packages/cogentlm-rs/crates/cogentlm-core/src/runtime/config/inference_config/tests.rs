//! Unit tests for the parent module.

use super::*;

#[test]
fn sampling_defaults_match_legacy_cpp_browser_runtime() {
    let sampling = SamplingRuntimeConfig::default();

    assert_eq!(
        sampling.samplers,
        vec![
            SamplerStage::TopK,
            SamplerStage::Penalties,
            SamplerStage::TopP,
            SamplerStage::Temperature,
        ]
    );
    assert_eq!(sampling.top_k, Some(40));
    assert_eq!(sampling.top_p, Some(0.8));
    assert_eq!(sampling.temperature, Some(0.7));
    assert_eq!(sampling.repeat_last_n, Some(64));
    assert_eq!(sampling.repeat_penalty, Some(1.05));
    assert_eq!(sampling.frequency_penalty, Some(0.0));
    assert_eq!(sampling.presence_penalty, Some(0.0));
    assert_eq!(sampling.backend_sampling, cfg!(not(target_arch = "wasm32")));
}

#[test]
fn native_runtime_config_deserializes_sparse_browser_json() {
    let config: NativeRuntimeConfig = serde_json::from_str(
        r#"{
            "placement": { "gpu_layers": { "count": 99 } },
            "context": { "n_ctx": 8192, "flash_attention": "enabled" },
            "sampling": {
                "samplers": ["top_k", "top_p", "temperature"],
                "typical_p": 0.95,
                "backend_sampling": true
            },
            "scheduler": {
                "policy": {
                    "mode": "throughput_first",
                    "decode_token_reserve": 2
                }
            }
        }"#,
    )
    .expect("browser runtime json");

    assert_eq!(config.placement.gpu_layers, GpuLayerConfig::Count(99));
    assert_eq!(config.context.n_ctx, Some(8192));
    assert_eq!(config.context.flash_attention, FlashAttentionMode::Enabled);
    assert_eq!(
        config.sampling.samplers,
        vec![
            SamplerStage::TopK,
            SamplerStage::TopP,
            SamplerStage::Temperature
        ]
    );
    assert_eq!(config.sampling.typical_p, Some(0.95));
    assert!(config.sampling.backend_sampling);
    assert_eq!(
        config.scheduler.policy.mode,
        SchedulerPolicyMode::ThroughputFirst
    );
    assert_eq!(config.scheduler.policy.decode_token_reserve, 2);
    assert!(!config.scheduler.policy.enable_adaptive_prefill_chunking);
}

#[test]
fn llama_common_args_are_normalized_at_public_boundary() {
    let config = NativeRuntimeConfig {
        context: ContextRuntimeConfig {
            n_ctx: Some(-1),
            n_batch: Some(0),
            n_ubatch: Some(-8),
            n_parallel: Some(0),
            n_threads: Some(-1),
            n_threads_batch: Some(-2),
            ..ContextRuntimeConfig::default()
        },
        ..NativeRuntimeConfig::default()
    };

    let args = config.llama_common_args();

    assert_arg_value(&args, "--ctx-size", "1");
    assert_arg_value(&args, "--batch-size", "1");
    assert_arg_value(&args, "--ubatch-size", "1");
    assert_arg_value(&args, "--parallel", "1");
    assert_arg_value(&args, "--threads", "0");
    assert_arg_value(&args, "--threads-batch", "0");
}

#[test]
fn llama_common_args_presizes_exact_argument_count() {
    let config = NativeRuntimeConfig {
        placement: ModelPlacementConfig {
            devices: vec!["gpu0".to_string(), "gpu1".to_string()],
            main_gpu: Some(1),
            tensor_split: vec![0.5, 0.5],
            fit_params_min_ctx: Some(2048),
            fit_params_target_bytes: vec![1024 * 1024],
            use_mlock: true,
            use_mmap: false,
            check_tensors: true,
            no_extra_bufts: true,
            no_host: true,
            ..ModelPlacementConfig::default()
        },
        context: ContextRuntimeConfig {
            n_ctx: Some(4096),
            n_batch: Some(512),
            n_ubatch: Some(128),
            n_threads: Some(8),
            n_threads_batch: Some(4),
            kv_unified: Some(true),
            swa_full: true,
            rope_scaling: Some(RopeScaling::Yarn),
            rope_freq_base: Some(10_000.0),
            rope_freq_scale: Some(1.0),
            yarn_orig_ctx: Some(4096),
            yarn_ext_factor: Some(1.0),
            yarn_attn_factor: Some(1.0),
            yarn_beta_fast: Some(32.0),
            yarn_beta_slow: Some(1.0),
            ..ContextRuntimeConfig::default()
        },
        ..NativeRuntimeConfig::default()
    };

    let args = config.llama_common_args();

    assert_eq!(args.capacity(), args.len());
}

#[test]
fn try_sampling_json_merges_overrides_without_silent_fallback() {
    let config = NativeRuntimeConfig::default();
    let override_config = SamplingRuntimeConfig {
        top_k: Some(12),
        backend_sampling: false,
        ..SamplingRuntimeConfig::default()
    };

    let json = config
        .try_sampling_json_with_override(Some(&override_config))
        .expect("sampling JSON");
    let value: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");

    assert_eq!(value["top_k"], 12);
    assert_eq!(value["backend_sampling"], false);
    assert_ne!(json, "{}");
}

fn assert_arg_value(args: &[String], key: &str, expected: &str) {
    let value = args
        .windows(2)
        .find_map(|window| (window[0] == key).then_some(window[1].as_str()));
    assert_eq!(value, Some(expected), "missing or wrong value for {key}");
}
