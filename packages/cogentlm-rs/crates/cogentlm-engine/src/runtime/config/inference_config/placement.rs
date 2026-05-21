use serde::{Deserialize, Serialize};

use super::{bool_arg, join_csv, push_arg};

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
            self.gpu_layers = GpuLayerConfig::Count(count.max(0));
        }
        self.main_gpu = self.main_gpu.map(|value| value.max(0));
        self.fit_params_min_ctx = self.fit_params_min_ctx.map(|value| value.max(1));
    }

    pub(super) fn arg_len(&self) -> usize {
        let mut len = 6;
        if !self.devices.is_empty() {
            len += 2;
        }
        if self.main_gpu.is_some() {
            len += 2;
        }
        if !self.tensor_split.is_empty() {
            len += 2;
        }
        if self.fit_params_min_ctx.is_some() {
            len += 2;
        }
        if !self.fit_params_target_bytes.is_empty() {
            len += 2;
        }
        if self.use_mlock {
            len += 1;
        }
        if !self.use_mmap {
            len += 1;
        }
        if self.check_tensors {
            len += 1;
        }
        if self.no_extra_bufts {
            len += 1;
        }
        if self.no_host {
            len += 1;
        }
        len
    }

    pub(super) fn push_args(&self, args: &mut Vec<String>) {
        if !self.devices.is_empty() {
            push_arg(args, "--device", self.devices.join(","));
        }
        match self.gpu_layers {
            GpuLayerConfig::Auto => push_arg(args, "--gpu-layers", "auto"),
            GpuLayerConfig::All => push_arg(args, "--gpu-layers", "all"),
            GpuLayerConfig::Count(count) => push_arg(args, "--gpu-layers", count.to_string()),
        }
        push_arg(args, "--split-mode", self.split_mode.as_llama_arg());
        if let Some(main_gpu) = self.main_gpu {
            push_arg(args, "--main-gpu", main_gpu.to_string());
        }
        if !self.tensor_split.is_empty() {
            push_arg(args, "--tensor-split", join_csv(self.tensor_split.iter()));
        }
        push_arg(args, "--fit", bool_arg(self.fit_params));
        if let Some(min_ctx) = self.fit_params_min_ctx {
            push_arg(args, "--fit-ctx", min_ctx.to_string());
        }
        if !self.fit_params_target_bytes.is_empty() {
            push_arg(
                args,
                "--fit-target",
                join_csv(
                    self.fit_params_target_bytes
                        .iter()
                        .map(|bytes| bytes / (1024 * 1024)),
                ),
            );
        }
        if self.use_mlock {
            args.push("--mlock".to_string());
        }
        if !self.use_mmap {
            args.push("--no-mmap".to_string());
        }
        if self.check_tensors {
            args.push("--check-tensors".to_string());
        }
        if self.no_extra_bufts {
            args.push("--no-repack".to_string());
        }
        if self.no_host {
            args.push("--no-host".to_string());
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuLayerConfig {
    Auto,
    All,
    Count(i32),
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
    use super::{GpuLayerConfig, ModelPlacementConfig, SplitMode};

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
            fit_params_target_bytes: vec![1024 * 1024],
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

    fn arg_value<'args>(args: &'args [String], key: &str) -> Option<&'args str> {
        args.windows(2)
            .find_map(|window| (window[0] == key).then_some(window[1].as_str()))
    }
}
