//! Tests the `runtime::config::inference_config::context::value_types` module
//! in `cogentlm`.
//!
//! Covers llama argument spellings and serde wire names for context value
//! enums through pure value assertions.

use serde_json::json;

use super::*;

#[test]
fn flash_attention_modes_map_to_llama_args_and_wire_names() {
    assert_eq!(FlashAttentionMode::default(), FlashAttentionMode::Auto);

    let cases = [
        (FlashAttentionMode::Auto, "auto", "auto"),
        (FlashAttentionMode::Enabled, "on", "enabled"),
        (FlashAttentionMode::Disabled, "off", "disabled"),
    ];

    for (mode, llama_arg, wire) in cases {
        assert_eq!(mode.as_llama_arg(), llama_arg);
        assert_eq!(serde_json::to_value(mode).expect("mode"), wire);
        assert_eq!(
            serde_json::from_value::<FlashAttentionMode>(json!(wire)).expect("mode"),
            mode
        );
    }
}

#[test]
fn kv_cache_types_map_to_llama_args_and_wire_names() {
    assert_eq!(KvCacheType::default(), KvCacheType::F16);

    let cases = [
        (KvCacheType::F16, "f16"),
        (KvCacheType::F32, "f32"),
        (KvCacheType::Q8_0, "q8_0"),
        (KvCacheType::Q4_0, "q4_0"),
        (KvCacheType::Q4_1, "q4_1"),
        (KvCacheType::Iq4Nl, "iq4_nl"),
        (KvCacheType::Q5_0, "q5_0"),
        (KvCacheType::Q5_1, "q5_1"),
    ];

    for (cache_type, wire) in cases {
        assert_eq!(cache_type.as_llama_arg(), wire);
        assert_eq!(serde_json::to_value(cache_type).expect("cache type"), wire);
        assert_eq!(
            serde_json::from_value::<KvCacheType>(json!(wire)).expect("cache type"),
            cache_type
        );
    }
}

#[test]
fn rope_scaling_modes_map_to_llama_args_and_wire_names() {
    let cases = [
        (RopeScaling::None, "none"),
        (RopeScaling::Linear, "linear"),
        (RopeScaling::Yarn, "yarn"),
    ];

    for (scaling, wire) in cases {
        assert_eq!(scaling.as_llama_arg(), wire);
        assert_eq!(serde_json::to_value(scaling).expect("rope scaling"), wire);
        assert_eq!(
            serde_json::from_value::<RopeScaling>(json!(wire)).expect("rope scaling"),
            scaling
        );
    }
}
