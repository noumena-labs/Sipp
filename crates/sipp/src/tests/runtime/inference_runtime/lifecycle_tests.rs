//! Tests the `runtime::inference_runtime::lifecycle` module in `sipp`.
//!
//! Covers deterministic inference-runtime helpers, state transitions, and error paths while avoiding native model execution unless a test is explicitly ignored.

use super::*;
use crate::engine::protocol::ModelClass;
use crate::error::Error;
use crate::native_bridge::NativeRuntimeHandle;
use crate::runtime::config::{NativeRuntimeConfig, ResolvedRuntimeLimits};
use crate::shard::GgufMetadataInspection;

fn profile(class: ModelClass) -> ModelProfile {
    ModelProfile {
        class,
        generates_audio: false,
    }
}

#[test]
fn encoder_only_enables_embedding_context_before_common_params() {
    let mut config = NativeRuntimeConfig::default();

    apply_model_requirements(&mut config, profile(ModelClass::EncoderOnly)).expect("defaults");

    assert_eq!(config.context.embeddings, Some(true));
    assert_eq!(config.context.n_batch, Some(DEFAULT_ENCODER_BATCH_SIZE));
    assert_eq!(config.context.n_ubatch, Some(DEFAULT_ENCODER_BATCH_SIZE));
}

#[test]
fn decoder_only_and_encoder_decoder_defaults_preserve_supported_configs() {
    let mut decoder_config = NativeRuntimeConfig::default();
    apply_model_requirements(&mut decoder_config, profile(ModelClass::DecoderOnly))
        .expect("decoder");
    assert_eq!(decoder_config.context.embeddings, None);

    let mut encoder_decoder_config = NativeRuntimeConfig::default();
    apply_model_requirements(
        &mut encoder_decoder_config,
        profile(ModelClass::EncoderDecoder),
    )
    .expect("encoder-decoder defaults");
    assert_eq!(encoder_decoder_config.context.embeddings, None);
    assert_eq!(
        encoder_decoder_config.context.n_batch,
        Some(DEFAULT_ENCODER_BATCH_SIZE)
    );
    assert_eq!(
        encoder_decoder_config.context.n_ubatch,
        Some(DEFAULT_ENCODER_BATCH_SIZE)
    );
    assert_eq!(encoder_decoder_config.context.n_parallel, Some(1));
}

#[test]
fn encoder_batch_defaults_follow_n_batch_and_preserve_explicit_n_ubatch() {
    let mut config = NativeRuntimeConfig::default();
    config.context.n_batch = Some(1024);

    apply_model_requirements(&mut config, profile(ModelClass::EncoderOnly))
        .expect("encoder defaults");

    assert_eq!(config.context.n_ubatch, Some(1024));

    let mut explicit_config = NativeRuntimeConfig::default();
    explicit_config.context.n_batch = Some(1024);
    explicit_config.context.n_ubatch = Some(256);

    apply_model_requirements(&mut explicit_config, profile(ModelClass::EncoderOnly))
        .expect("explicit encoder defaults");

    assert_eq!(explicit_config.context.n_batch, Some(1024));
    assert_eq!(explicit_config.context.n_ubatch, Some(256));
}

#[test]
fn encoder_decoder_rejects_embedding_context_before_common_params() {
    let mut config = NativeRuntimeConfig::default();
    config.context.embeddings = Some(true);

    let error = apply_model_requirements(&mut config, profile(ModelClass::EncoderDecoder))
        .expect_err("encoder-decoder embeddings");

    assert!(
        matches!(error, Error::UnsupportedOperation { operation: "load", reason }
            if reason.contains("embedding output"))
    );
}

#[test]
fn runtime_parts_new_allocates_minimum_scheduler_and_batch_state() {
    let config = NativeRuntimeConfig::default();
    let parts =
        RuntimeParts::new(&config, ResolvedRuntimeLimits::default()).expect("runtime parts");

    assert_eq!(parts.max_sequences, 1);
    assert_eq!(parts.slot_scheduler.slots.len(), 1);
    assert!(parts.scratch_token_capacity >= 1);
}

#[test]
fn empty_native_runtime_reports_missing_model_class() {
    let runtime = NativeRuntimeHandle::empty_for_tests();

    let error = model_class_from_init(&runtime).expect_err("model class");

    assert!(matches!(
        error,
        Error::UnsupportedOperation { operation: "load", reason }
            if reason.contains("neither encoder nor decoder")
    ));
}

#[test]
fn init_multimodal_context_without_projector_is_noop_and_projector_needs_native_context() {
    let mut runtime = NativeRuntimeHandle::empty_for_tests();
    let config = NativeRuntimeConfig::default();

    init_multimodal_context(&config, &mut runtime).expect("no projector");

    let mut config = NativeRuntimeConfig::default();
    config.multimodal.projector_path = Some("projector.gguf".into());
    let error = init_multimodal_context(&config, &mut runtime).expect_err("projector init");

    assert!(matches!(
        error,
        Error::NullPointer("sipp_mtmd_init_from_file")
    ));
}

#[test]
fn encoder_decoder_rejects_parallel_contexts_before_common_params() {
    let mut config = NativeRuntimeConfig::default();
    config.context.n_parallel = Some(2);

    let error = apply_model_requirements(&mut config, profile(ModelClass::EncoderDecoder))
        .expect_err("encoder-decoder parallelism");

    assert!(
        matches!(error, Error::UnsupportedOperation { operation: "load", reason }
            if reason.contains("n_parallel=1"))
    );
}

#[test]
fn audio_generation_requires_embeddings_and_one_sequence() {
    let mut config = NativeRuntimeConfig::default();
    apply_model_requirements(
        &mut config,
        ModelProfile {
            class: ModelClass::DecoderOnly,
            generates_audio: true,
        },
    )
    .expect("audio-generation requirements");

    assert_eq!(config.context.embeddings, Some(true));
    assert_eq!(config.context.n_parallel, Some(1));

    config.context.n_parallel = Some(2);
    let error = apply_model_requirements(
        &mut config,
        ModelProfile {
            class: ModelClass::DecoderOnly,
            generates_audio: true,
        },
    )
    .expect_err("parallel audio generation");
    assert!(matches!(
        error,
        Error::UnsupportedOperation {
            operation: "load",
            ..
        }
    ));
}

#[test]
fn audio_generation_requirement_comes_from_projector_capability_metadata() {
    let mut metadata = GgufMetadataInspection::default();
    assert!(!metadata_generates_audio(&metadata));

    metadata.clip_has_audio_generation_encoder = Some(true);
    assert!(metadata_generates_audio(&metadata));

    metadata.clip_has_audio_generation_encoder = Some(false);
    metadata.clip_audio_generation_projector_type = Some("future_audio_generator".to_string());
    assert!(metadata_generates_audio(&metadata));
}
