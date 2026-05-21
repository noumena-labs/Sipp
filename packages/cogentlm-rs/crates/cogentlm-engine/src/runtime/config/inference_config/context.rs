use serde::{Deserialize, Serialize};

use super::{positive_or_none, push_arg};

mod value_types;

pub use value_types::{FlashAttentionMode, KvCacheType, RopeScaling};

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
        }
    }
}

impl ContextRuntimeConfig {
    pub(super) fn normalize(&mut self) {
        self.n_ctx = positive_or_none(self.n_ctx, 1);
        self.n_batch = positive_or_none(self.n_batch, 1);
        self.n_ubatch = positive_or_none(self.n_ubatch, 1);
        self.n_parallel = Some(self.n_parallel.unwrap_or(1).max(1));
        self.n_threads = positive_or_none(self.n_threads, 0);
        self.n_threads_batch = positive_or_none(self.n_threads_batch, 0);
    }

    pub(super) fn push_args(&self, args: &mut Vec<String>) {
        if let Some(value) = self.n_ctx {
            push_arg(args, "--ctx-size", value.to_string());
        }
        if let Some(value) = self.n_batch {
            push_arg(args, "--batch-size", value.to_string());
        }
        if let Some(value) = self.n_ubatch {
            push_arg(args, "--ubatch-size", value.to_string());
        }
        if let Some(value) = self.n_parallel {
            push_arg(args, "--parallel", value.to_string());
        }
        if let Some(value) = self.n_threads {
            push_arg(args, "--threads", value.to_string());
        }
        if let Some(value) = self.n_threads_batch {
            push_arg(args, "--threads-batch", value.to_string());
        }
        push_arg(args, "--flash-attn", self.flash_attention.as_llama_arg());
        if let Some(value) = self.kv_unified {
            args.push(
                if value {
                    "--kv-unified"
                } else {
                    "--no-kv-unified"
                }
                .to_string(),
            );
        }
        push_arg(args, "--cache-type-k", self.cache_type_k.as_llama_arg());
        push_arg(args, "--cache-type-v", self.cache_type_v.as_llama_arg());
        args.push(
            if self.offload_kqv {
                "--kv-offload"
            } else {
                "--no-kv-offload"
            }
            .to_string(),
        );
        args.push(
            if self.op_offload {
                "--op-offload"
            } else {
                "--no-op-offload"
            }
            .to_string(),
        );
        if self.swa_full {
            args.push("--swa-full".to_string());
        }
        args.push(
            if self.warmup {
                "--warmup"
            } else {
                "--no-warmup"
            }
            .to_string(),
        );
        if let Some(value) = self.rope_scaling {
            push_arg(args, "--rope-scaling", value.as_llama_arg());
        }
        if let Some(value) = self.rope_freq_base {
            push_arg(args, "--rope-freq-base", value.to_string());
        }
        if let Some(value) = self.rope_freq_scale {
            push_arg(args, "--rope-freq-scale", value.to_string());
        }
        if let Some(value) = self.yarn_orig_ctx {
            push_arg(args, "--yarn-orig-ctx", value.to_string());
        }
        if let Some(value) = self.yarn_ext_factor {
            push_arg(args, "--yarn-ext-factor", value.to_string());
        }
        if let Some(value) = self.yarn_attn_factor {
            push_arg(args, "--yarn-attn-factor", value.to_string());
        }
        if let Some(value) = self.yarn_beta_fast {
            push_arg(args, "--yarn-beta-fast", value.to_string());
        }
        if let Some(value) = self.yarn_beta_slow {
            push_arg(args, "--yarn-beta-slow", value.to_string());
        }
    }

    pub(super) fn arg_len(&self) -> usize {
        let mut len = 9;
        len += self.n_ctx.map_or(0, |_| 2);
        len += self.n_batch.map_or(0, |_| 2);
        len += self.n_ubatch.map_or(0, |_| 2);
        len += self.n_parallel.map_or(0, |_| 2);
        len += self.n_threads.map_or(0, |_| 2);
        len += self.n_threads_batch.map_or(0, |_| 2);
        len += self.kv_unified.map_or(0, |_| 1);
        len += usize::from(self.swa_full);
        len += self.rope_scaling.map_or(0, |_| 2);
        len += self.rope_freq_base.map_or(0, |_| 2);
        len += self.rope_freq_scale.map_or(0, |_| 2);
        len += self.yarn_orig_ctx.map_or(0, |_| 2);
        len += self.yarn_ext_factor.map_or(0, |_| 2);
        len += self.yarn_attn_factor.map_or(0, |_| 2);
        len += self.yarn_beta_fast.map_or(0, |_| 2);
        len += self.yarn_beta_slow.map_or(0, |_| 2);
        len
    }
}

#[cfg(test)]
mod tests {
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

    fn arg_value<'args>(args: &'args [String], key: &str) -> Option<&'args str> {
        args.windows(2)
            .find_map(|window| (window[0] == key).then_some(window[1].as_str()))
    }
}
