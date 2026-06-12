use serde::{Deserialize, Serialize};

use super::{
    args_len, positive_or_default, positive_or_none, push_arg, push_flag, push_flag_pair,
    push_optional_arg,
};
use crate::engine::protocol::PoolingType;

mod value_types;

pub use value_types::{FlashAttentionMode, KvCacheType, RopeScaling};

const ALWAYS_EMITTED_KEY_VALUE_ARGS: usize = 3;
const ALWAYS_EMITTED_FLAGS: usize = 3;
const BASE_ARG_LEN: usize =
    ALWAYS_EMITTED_KEY_VALUE_ARGS * super::KEY_VALUE_ARG_LEN + ALWAYS_EMITTED_FLAGS;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ContextRuntimeConfig {
    pub n_ctx: Option<i32>,
    pub n_batch: Option<i32>,
    pub n_ubatch: Option<i32>,
    pub n_parallel: Option<i32>,
    pub n_threads: Option<i32>,
    pub n_threads_batch: Option<i32>,
    pub flash_attention: FlashAttentionMode,
    pub kv_unified: Option<bool>,
    pub cache_type_k: KvCacheType,
    pub cache_type_v: KvCacheType,
    pub offload_kqv: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub warmup: bool,
    pub rope_scaling: Option<RopeScaling>,
    pub rope_freq_base: Option<f32>,
    pub rope_freq_scale: Option<f32>,
    pub yarn_orig_ctx: Option<i32>,
    pub yarn_ext_factor: Option<f32>,
    pub yarn_attn_factor: Option<f32>,
    pub yarn_beta_fast: Option<f32>,
    pub yarn_beta_slow: Option<f32>,
    /// Map to `--embedding`. When `Some(true)`, the llama context is created
    /// with embedding output enabled. Encoder-only models pick this up
    /// automatically at load; decoder-only callers (e.g. Qwen-embedding) set
    /// it explicitly.
    pub embeddings: Option<bool>,
    /// Map to `--pooling none|mean|cls|last|rank`. `None` (or `Unspecified`)
    /// defers to the model's `<arch>.pooling_type` GGUF metadata.
    pub pooling: Option<PoolingType>,
}

impl Default for ContextRuntimeConfig {
    fn default() -> Self {
        Self {
            n_ctx: None,
            n_batch: None,
            n_ubatch: None,
            n_parallel: Some(1),
            n_threads: None,
            n_threads_batch: None,
            flash_attention: FlashAttentionMode::Auto,
            kv_unified: None,
            cache_type_k: KvCacheType::F16,
            cache_type_v: KvCacheType::F16,
            offload_kqv: true,
            op_offload: true,
            swa_full: false,
            warmup: true,
            rope_scaling: None,
            rope_freq_base: None,
            rope_freq_scale: None,
            yarn_orig_ctx: None,
            yarn_ext_factor: None,
            yarn_attn_factor: None,
            yarn_beta_fast: None,
            yarn_beta_slow: None,
            embeddings: None,
            pooling: None,
        }
    }
}

impl ContextRuntimeConfig {
    pub(super) fn normalize(&mut self) {
        self.n_ctx = positive_or_none(self.n_ctx, 1);
        self.n_batch = positive_or_none(self.n_batch, 1);
        self.n_ubatch = positive_or_none(self.n_ubatch, 1);
        self.n_parallel = Some(positive_or_default(self.n_parallel, 1, 1));
        self.n_threads = positive_or_none(self.n_threads, 0);
        self.n_threads_batch = positive_or_none(self.n_threads_batch, 0);
    }

    pub(super) fn push_args(&self, args: &mut Vec<String>) {
        push_optional_arg(args, "--ctx-size", self.n_ctx);
        push_optional_arg(args, "--batch-size", self.n_batch);
        push_optional_arg(args, "--ubatch-size", self.n_ubatch);
        push_optional_arg(args, "--parallel", self.n_parallel);
        push_optional_arg(args, "--threads", self.n_threads);
        push_optional_arg(args, "--threads-batch", self.n_threads_batch);
        push_arg(args, "--flash-attn", self.flash_attention.as_llama_arg());
        if let Some(value) = self.kv_unified {
            push_flag_pair(args, value, "--kv-unified", "--no-kv-unified");
        }
        push_arg(args, "--cache-type-k", self.cache_type_k.as_llama_arg());
        push_arg(args, "--cache-type-v", self.cache_type_v.as_llama_arg());
        push_flag_pair(args, self.offload_kqv, "--kv-offload", "--no-kv-offload");
        push_flag_pair(args, self.op_offload, "--op-offload", "--no-op-offload");
        push_flag(args, "--swa-full", self.swa_full);
        push_flag_pair(args, self.warmup, "--warmup", "--no-warmup");
        if let Some(value) = self.rope_scaling {
            push_arg(args, "--rope-scaling", value.as_llama_arg());
        }
        push_optional_arg(args, "--rope-freq-base", self.rope_freq_base);
        push_optional_arg(args, "--rope-freq-scale", self.rope_freq_scale);
        push_optional_arg(args, "--yarn-orig-ctx", self.yarn_orig_ctx);
        push_optional_arg(args, "--yarn-ext-factor", self.yarn_ext_factor);
        push_optional_arg(args, "--yarn-attn-factor", self.yarn_attn_factor);
        push_optional_arg(args, "--yarn-beta-fast", self.yarn_beta_fast);
        push_optional_arg(args, "--yarn-beta-slow", self.yarn_beta_slow);
        if self.embeddings == Some(true) {
            push_flag(args, "--embedding", true);
        }
        if let Some(pooling) = self.pooling {
            if pooling.is_explicit() {
                push_arg(args, "--pooling", pooling.as_str());
            }
        }
    }

    pub(super) fn arg_len(&self) -> usize {
        args_len(
            BASE_ARG_LEN,
            [
                self.n_ctx.is_some(),
                self.n_batch.is_some(),
                self.n_ubatch.is_some(),
                self.n_parallel.is_some(),
                self.n_threads.is_some(),
                self.n_threads_batch.is_some(),
                self.rope_scaling.is_some(),
                self.rope_freq_base.is_some(),
                self.rope_freq_scale.is_some(),
                self.yarn_orig_ctx.is_some(),
                self.yarn_ext_factor.is_some(),
                self.yarn_attn_factor.is_some(),
                self.yarn_beta_fast.is_some(),
                self.yarn_beta_slow.is_some(),
                self.pooling.is_some_and(PoolingType::is_explicit),
            ],
            [
                self.kv_unified.is_some(),
                self.swa_full,
                self.embeddings == Some(true),
            ],
        )
    }
}

#[cfg(test)]
#[path = "../../../tests/runtime/config/inference_config/context_tests.rs"]
mod context_tests;
