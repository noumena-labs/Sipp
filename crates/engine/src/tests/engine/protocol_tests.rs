//! Tests the `engine::protocol` module in `cogentlm-engine`.
//!
//! Covers engine public values and helper behavior with deterministic unit fixtures; model-backed checks stay explicitly ignored.

use super::*;

#[test]
fn engine_status_strings_cover_all_variants() {
    let cases = [
        (EngineStatus::Idle, "idle"),
        (EngineStatus::Loading, "loading"),
        (EngineStatus::Ready, "ready"),
        (EngineStatus::Running, "running"),
        (EngineStatus::Error, "error"),
        (EngineStatus::Closed, "closed"),
    ];

    for (status, expected) in cases {
        assert_eq!(status.as_str(), expected);
    }
}

#[test]
fn request_status_strings_cover_all_variants() {
    let cases = [
        (RequestStatus::Queued, "queued"),
        (RequestStatus::Prefill, "prefill"),
        (RequestStatus::Decode, "decode"),
        (RequestStatus::Completed, "completed"),
        (RequestStatus::Failed, "failed"),
        (RequestStatus::Cancelled, "cancelled"),
    ];

    for (status, expected) in cases {
        assert_eq!(status.as_str(), expected);
    }
}

#[test]
fn pooling_type_maps_names_llama_values_and_explicitness() {
    let cases = [
        (PoolingType::Unspecified, "unspecified", -1, false),
        (PoolingType::None, "none", 0, true),
        (PoolingType::Mean, "mean", 1, true),
        (PoolingType::Cls, "cls", 2, true),
        (PoolingType::Last, "last", 3, true),
        (PoolingType::Rank, "rank", 4, true),
    ];

    for (pooling, name, llama_value, explicit) in cases {
        assert_eq!(pooling.as_str(), name);
        assert_eq!(PoolingType::from_name(name), Some(pooling));
        assert_eq!(PoolingType::from_llama_value(llama_value), Some(pooling));
        assert_eq!(pooling.is_explicit(), explicit);
    }
    assert_eq!(PoolingType::from_name("bad"), None);
    assert_eq!(PoolingType::from_llama_value(99), None);
}

#[test]
fn model_class_maps_known_architectures_and_defaults_decoder_only() {
    assert_eq!(
        ModelClass::from_architecture("bert"),
        ModelClass::EncoderOnly
    );
    assert_eq!(
        ModelClass::from_architecture("t5encoder"),
        ModelClass::EncoderOnly
    );
    assert_eq!(
        ModelClass::from_architecture("t5"),
        ModelClass::EncoderDecoder
    );
    assert_eq!(
        ModelClass::from_architecture("qwen2"),
        ModelClass::DecoderOnly
    );

    assert_eq!(ModelClass::DecoderOnly.as_str(), "decoder_only");
    assert_eq!(ModelClass::EncoderDecoder.as_str(), "encoder_decoder");
    assert_eq!(ModelClass::EncoderOnly.as_str(), "encoder_only");
}

#[test]
fn embed_options_and_engine_state_defaults_are_stable() {
    let options = EmbedOptions::default();
    assert!(options.normalize);
    assert_eq!(options.context_key, None);

    let state = EngineState::default();
    assert_eq!(state.status, EngineStatus::Idle);
    assert_eq!(state.model, None);
    assert_eq!(state.backend, BackendInfo::default());
    assert_eq!(state.runtime, None);
    assert!(state.requests.is_empty());
    assert_eq!(state.stats, EngineStats::default());
    assert_eq!(state.updated_at_unix_ms, 0);
}
