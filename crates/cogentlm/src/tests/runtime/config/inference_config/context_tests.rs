//! Tests the `runtime::config::inference_config::context` module in `cogentlm`.
//!
//! Covers runtime configuration normalization, serialization, and boundary choices through pure value assertions.

use super::super::arg_value;
use super::{ContextRuntimeConfig, FlashAttentionMode, KvCacheType, RopeScaling};

#[test]
fn context_normalize_clamps_thread_and_batch_limits() {
    let mut context = ContextRuntimeConfig {
        n_ctx: Some(-1),
        n_batch: Some(0),
        n_ubatch: Some(-8),
        n_parallel: Some(0),
        n_threads: Some(-1),
        n_threads_batch: Some(-2),
        ..ContextRuntimeConfig::default()
    };

    context.normalize();

    assert_eq!(context.n_ctx, Some(1));
    assert_eq!(context.n_batch, Some(1));
    assert_eq!(context.n_ubatch, Some(1));
    assert_eq!(context.n_parallel, Some(1));
    assert_eq!(context.n_threads, Some(0));
    assert_eq!(context.n_threads_batch, Some(0));
}

#[test]
fn context_arg_len_matches_emitted_args() {
    let context = ContextRuntimeConfig {
        n_ctx: Some(4096),
        n_batch: Some(512),
        n_ubatch: Some(128),
        n_threads: Some(8),
        n_threads_batch: Some(4),
        kv_unified: Some(true),
        flash_attention: FlashAttentionMode::Enabled,
        cache_type_k: KvCacheType::Q8_0,
        cache_type_v: KvCacheType::Q4_0,
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
    };
    let mut args = Vec::with_capacity(context.arg_len());

    context.push_args(&mut args);

    assert_eq!(args.capacity(), args.len());
    assert_eq!(arg_value(&args, "--ctx-size"), Some("4096"));
    assert_eq!(arg_value(&args, "--flash-attn"), Some("on"));
    assert_eq!(arg_value(&args, "--cache-type-k"), Some("q8_0"));
    assert!(args.iter().any(|arg| arg == "--kv-unified"));
}
