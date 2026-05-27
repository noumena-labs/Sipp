use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::defaults::BYTES_PER_MIB_U64;

use super::{args_len, positive_or_none, push_arg, push_csv_arg, push_flag, push_optional_arg};

const ALWAYS_EMITTED_KEY_VALUE_ARGS: usize = 3;
const BASE_ARG_LEN: usize = ALWAYS_EMITTED_KEY_VALUE_ARGS * super::KEY_VALUE_ARG_LEN;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelPlacementConfig {
    pub devices: Vec<String>,
    pub gpu_layers: GpuLayerConfig,
    pub split_mode: SplitMode,
    pub main_gpu: Option<i32>,
    pub tensor_split: Vec<f32>,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub fit_params: bool,
    pub fit_params_min_ctx: Option<i32>,
    pub fit_params_target_bytes: Vec<u64>,
    pub check_tensors: bool,
    pub no_extra_bufts: bool,
    pub no_host: bool,
}

impl Default for ModelPlacementConfig {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            gpu_layers: GpuLayerConfig::Auto,
            split_mode: SplitMode::Layer,
            main_gpu: None,
            tensor_split: Vec::new(),
            use_mmap: cfg!(not(target_family = "wasm")),
            use_mlock: false,
            fit_params: false,
            fit_params_min_ctx: None,
            fit_params_target_bytes: Vec::new(),
            check_tensors: false,
            no_extra_bufts: false,
            no_host: false,
        }
    }
}

impl ModelPlacementConfig {
    pub(super) fn normalize(&mut self) {
        if let GpuLayerConfig::Count(count) = self.gpu_layers {
            self.gpu_layers = GpuLayerConfig::from_layer_count(count);
        }
        self.main_gpu = positive_or_none(self.main_gpu, 0);
        self.fit_params_min_ctx = positive_or_none(self.fit_params_min_ctx, 1);
    }

    pub(super) fn arg_len(&self) -> usize {
        args_len(
            BASE_ARG_LEN,
            [
                !self.devices.is_empty(),
                self.main_gpu.is_some(),
                !self.tensor_split.is_empty(),
                self.fit_params_min_ctx.is_some(),
                !self.fit_params_target_bytes.is_empty(),
            ],
            [
                self.use_mlock,
                !self.use_mmap,
                self.check_tensors,
                self.no_extra_bufts,
                self.no_host,
            ],
        )
    }

    pub(super) fn push_args(&self, args: &mut Vec<String>) {
        if !self.devices.is_empty() {
            push_csv_arg(args, "--device", self.devices.iter());
        }
        push_arg(args, "--gpu-layers", self.gpu_layers.to_llama_arg());
        push_arg(args, "--split-mode", self.split_mode.as_llama_arg());
        push_optional_arg(args, "--main-gpu", self.main_gpu);
        if !self.tensor_split.is_empty() {
            push_csv_arg(args, "--tensor-split", self.tensor_split.iter());
        }
        push_arg(args, "--fit", if self.fit_params { "on" } else { "off" });
        push_optional_arg(args, "--fit-ctx", self.fit_params_min_ctx);
        if !self.fit_params_target_bytes.is_empty() {
            push_csv_arg(
                args,
                "--fit-target",
                self.fit_params_target_bytes
                    .iter()
                    .map(|bytes| bytes / BYTES_PER_MIB_U64),
            );
        }
        push_flag(args, "--mlock", self.use_mlock);
        push_flag(args, "--no-mmap", !self.use_mmap);
        push_flag(args, "--check-tensors", self.check_tensors);
        push_flag(args, "--no-repack", self.no_extra_bufts);
        push_flag(args, "--no-host", self.no_host);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuLayerConfig {
    Auto,
    All,
    Count(i32),
}

impl GpuLayerConfig {
    pub fn from_layer_count(count: i32) -> Self {
        if count < 0 {
            Self::All
        } else {
            Self::Count(count)
        }
    }

    fn to_llama_arg(self) -> Cow<'static, str> {
        match self {
            Self::Auto => Cow::Borrowed("auto"),
            Self::All => Cow::Borrowed("all"),
            Self::Count(count) => Cow::Owned(count.to_string()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitMode {
    None,
    Layer,
    Row,
    Tensor,
}

impl SplitMode {
    fn as_llama_arg(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Layer => "layer",
            Self::Row => "row",
            Self::Tensor => "tensor",
        }
    }
}

#[cfg(test)]
mod tests {
    mod placement_tests;
}
