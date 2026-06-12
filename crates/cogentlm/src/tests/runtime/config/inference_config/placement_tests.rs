//! Tests the `runtime::config::inference_config::placement` module in `cogentlm`.
//!
//! Covers runtime configuration normalization, serialization, and boundary choices through pure value assertions.

use super::super::arg_value;
use super::{GpuLayerConfig, ModelPlacementConfig, SplitMode};
use crate::defaults::BYTES_PER_MIB_U64;

#[test]
fn gpu_layer_count_matches_llama_negative_all_convention() {
    assert_eq!(GpuLayerConfig::from_layer_count(-1), GpuLayerConfig::All);
    assert_eq!(
        GpuLayerConfig::from_layer_count(0),
        GpuLayerConfig::Count(0)
    );
    assert_eq!(
        GpuLayerConfig::from_layer_count(1),
        GpuLayerConfig::Count(1)
    );
}

#[test]
fn placement_arg_len_matches_emitted_args() {
    let placement = ModelPlacementConfig {
        devices: vec!["gpu0".to_string(), "gpu1".to_string()],
        gpu_layers: GpuLayerConfig::Count(99),
        split_mode: SplitMode::Tensor,
        main_gpu: Some(1),
        tensor_split: vec![0.5, 0.5],
        use_mlock: true,
        use_mmap: false,
        fit_params: true,
        fit_params_min_ctx: Some(2048),
        fit_params_target_bytes: vec![BYTES_PER_MIB_U64],
        check_tensors: true,
        no_extra_bufts: true,
        no_host: true,
    };
    let mut args = Vec::with_capacity(placement.arg_len());

    placement.push_args(&mut args);

    assert_eq!(args.capacity(), args.len());
    assert_eq!(arg_value(&args, "--device"), Some("gpu0,gpu1"));
    assert_eq!(arg_value(&args, "--gpu-layers"), Some("99"));
    assert_eq!(arg_value(&args, "--split-mode"), Some("tensor"));
    assert!(args.iter().any(|arg| arg == "--no-mmap"));
    assert!(args.iter().any(|arg| arg == "--no-host"));
}
