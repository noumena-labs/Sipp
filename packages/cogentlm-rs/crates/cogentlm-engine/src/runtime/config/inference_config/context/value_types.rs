use serde::{Deserialize, Serialize};

use crate::choice::choice_from_aliases;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum FlashAttentionMode {
    #[default]
    Auto,
    Enabled,
    Disabled,
}

impl FlashAttentionMode {
    pub fn from_choice(value: &str) -> Option<Self> {
        choice_from_aliases(
            value,
            &[
                (&["auto"], Self::Auto),
                (&["enabled", "enable", "on", "true"], Self::Enabled),
                (&["disabled", "disable", "off", "false"], Self::Disabled),
            ],
        )
    }

    pub(super) fn as_llama_arg(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Enabled => "on",
            Self::Disabled => "off",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheType {
    #[default]
    F16,
    F32,
    Q8_0,
    Q4_0,
    Q4_1,
    Iq4Nl,
    Q5_0,
    Q5_1,
}

impl KvCacheType {
    pub fn from_choice(value: &str) -> Option<Self> {
        choice_from_aliases(
            value,
            &[
                (&["f16"], Self::F16),
                (&["f32"], Self::F32),
                (&["q8_0"], Self::Q8_0),
                (&["q4_0"], Self::Q4_0),
                (&["q4_1"], Self::Q4_1),
                (&["iq4_nl"], Self::Iq4Nl),
                (&["q5_0"], Self::Q5_0),
                (&["q5_1"], Self::Q5_1),
            ],
        )
    }

    pub(super) fn as_llama_arg(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Iq4Nl => "iq4_nl",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeScaling {
    None,
    Linear,
    Yarn,
}

impl RopeScaling {
    pub fn from_choice(value: &str) -> Option<Self> {
        choice_from_aliases(
            value,
            &[
                (&["none"], Self::None),
                (&["linear"], Self::Linear),
                (&["yarn"], Self::Yarn),
            ],
        )
    }

    pub(super) fn as_llama_arg(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear => "linear",
            Self::Yarn => "yarn",
        }
    }
}
